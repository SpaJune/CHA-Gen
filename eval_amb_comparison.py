import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from eval_amb_identification import (
    apply_chat_template,
    get_sampling_params,
    process_outputs,
    save_json,
    set_global_seed,
)


PROMPT_TEMPLATES = {
    "default": (
        "下面有两个句子/词组，请判断哪个更容易引起歧义。只回答1或2。"
        "\n句子1：{sent_1}\n句子2：{sent_2}"
    ),
    "cot": (
        "下面有两个句子/词组，请判断哪个更容易引起歧义。"
        "请逐步思考，写出你的思考过程，在最后一行单独写出结论“1”或“2”。"
        "\n句子1：{sent_1}\n句子2：{sent_2}"
    ),
}

CHOICES = ("1", "2")


def get_prompt_template(style: str = "default") -> str:
    try:
        return PROMPT_TEMPLATES[style]
    except KeyError as exc:
        supported = ", ".join(PROMPT_TEMPLATES)
        raise ValueError(
            f"Unsupported prompt style: {style}. Choose from: {supported}"
        ) from exc


def load_dataset(name: str) -> list[dict[str, str]]:
    if name == "CHA-Gen":
        data = pd.read_csv("dataset/CHA-Gen/sentence_pair.csv")
        required_columns = {"ambiguous", "unambiguous"}
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(
                f"Dataset is missing columns: {sorted(missing_columns)}"
            )

        return data.loc[:, ["ambiguous", "unambiguous"]].astype(str).to_dict(
            orient="records"
        )
    else:
        raise ValueError(f"Unsupported dataset: {name}")


def arrange_pair(
    ambiguous: str,
    unambiguous: str,
) -> list[tuple[str, str, str, str]]:
    return [
        (ambiguous, unambiguous, "1", "original"),
        (unambiguous, ambiguous, "2", "reversed"),
    ]


def run_comparison(args: argparse.Namespace) -> Path:
    from vllm import LLM

    sentence_pairs = load_dataset(args.dataset)
    prompt_template = get_prompt_template(args.prompt_style)
    sampling_params, sampling_config = get_sampling_params(args)

    model = LLM(
        model=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )
    tokenizer = model.get_tokenizer()

    model_name = args.model_id.rstrip("/").split("/")[-1]
    thinking_suffix = "_thinking" if args.thinking else ""
    output_path = (
        Path(args.output_dir)
        / args.dataset
        / args.prompt_style
        / f"{model_name}{thinking_suffix}.json"
    )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite to replace it"
        )

    report = {
        "model_id": args.model_id,
        "dataset": args.dataset,
        "prompt_style": args.prompt_style,
        "prompt_template": prompt_template,
        "thinking": args.thinking,
        "evaluate_both_orders": True,
        "sampling_config": sampling_config,
        "results": [],
    }

    for start in tqdm(
        range(0, len(sentence_pairs), args.batch_size),
        desc="Comparing ambiguity",
    ):
        batch_pairs = sentence_pairs[start : start + args.batch_size]
        arranged_pairs = []
        for batch_offset, pair in enumerate(batch_pairs):
            pair_index = start + batch_offset
            for sent_1, sent_2, expected_answer, order in arrange_pair(
                pair["ambiguous"],
                pair["unambiguous"],
            ):
                arranged_pairs.append(
                    {
                        "pair_index": pair_index,
                        "ambiguous": pair["ambiguous"],
                        "unambiguous": pair["unambiguous"],
                        "sentence_1": sent_1,
                        "sentence_2": sent_2,
                        "expected_answer": expected_answer,
                        "order": order,
                    }
                )
        prompts = [
            prompt_template.format(
                sent_1=pair["sentence_1"],
                sent_2=pair["sentence_2"],
            )
            for pair in arranged_pairs
        ]
        input_texts = apply_chat_template(
            tokenizer,
            prompts,
            args.model_id,
            args.thinking,
        )
        outputs = model.generate(input_texts, sampling_params)
        predictions = process_outputs(
            outputs,
            input_texts,
            tokenizer,
            args.prompt_style,
            args.thinking,
            choices=CHOICES,
        )

        for pair, prediction in zip(
            arranged_pairs,
            predictions,
        ):
            report["results"].append(
                {
                    "pair_index": pair["pair_index"],
                    "order": pair["order"],
                    "ambiguous_sentence": pair["ambiguous"],
                    "unambiguous_sentence": pair["unambiguous"],
                    "sentence_1": pair["sentence_1"],
                    "sentence_2": pair["sentence_2"],
                    "expected_answer": pair["expected_answer"],
                    **prediction,
                }
            )
        save_json(output_path, report)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pairwise ambiguity comparison with vLLM."
    )
    parser.add_argument("--model_id", required=True, help="Hugging Face model ID")
    parser.add_argument("--dataset", default="CHA-Gen")
    parser.add_argument(
        "--prompt_style",
        choices=sorted(PROMPT_TEMPLATES),
        default="default",
    )
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--output_dir", default="results/comparison")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument(
        "--logprobs",
        type=int,
        default=20,
        help="Number of top next-token logprobs retained by vLLM",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.prompt_style == "cot" and args.thinking:
        parser.error("--prompt_style cot and --thinking cannot be used together")
    return args


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    output_path = run_comparison(args)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
