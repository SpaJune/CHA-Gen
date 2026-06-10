import argparse
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


PROMPT_TEMPLATES = {
    "default": (
        "请判断下面的句子/词组是否具有歧义。只回答“是”或“否”。"
        "\n\n句子：\n{sentence}"
    ),
    "cot": (
        "请判断下面的句子/词组是否具有歧义。请逐步思考，写出你的思考过程，"
        "在最后一行单独写出结论“是”或“否”。\n\n句子：\n{sentence}"
    ),
}

CHOICES = ("是", "否")
LABEL_TO_ANSWER = {1: "是", 0: "否"}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_prompt_template(style: str = "default") -> str:
    try:
        return PROMPT_TEMPLATES[style]
    except KeyError as exc:
        supported = ", ".join(PROMPT_TEMPLATES)
        raise ValueError(
            f"Unsupported prompt style: {style}. Choose from: {supported}"
        ) from exc


def load_dataset(name: str) -> tuple[list[str], list[int]]:
    if name == "CHA-Gen":
        data = pd.read_csv("dataset/CHA-Gen/corpus.csv")
        required_columns = {"sent", "ambiguity"}
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(
                f"Dataset is missing columns: {sorted(missing_columns)}"
            )

        sentences = data["sent"].astype(str).tolist()
        labels = data["ambiguity"].astype(int).tolist()
        invalid_labels = sorted(set(labels) - set(LABEL_TO_ANSWER))
        if invalid_labels:
            raise ValueError(f"Unexpected ambiguity labels: {invalid_labels}")
        return sentences, labels
    else:
        raise ValueError(f"Unsupported dataset: {name}")


def apply_chat_template(
    tokenizer: Any,
    prompts: list[str],
    model_id: str,
    thinking: bool,
) -> list[str]:
    input_texts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if "qwen3" in model_id.lower():
            template_kwargs["enable_thinking"] = thinking
        elif thinking:
            raise ValueError(
                "--thinking is currently supported only for Qwen3 models"
            )
        input_texts.append(
            tokenizer.apply_chat_template(messages, **template_kwargs)
        )
    return input_texts


def decoded_token(logprob: Any, tokenizer: Any, token_id: int) -> str:
    token = getattr(logprob, "decoded_token", None)
    return token if token is not None else tokenizer.decode([token_id])


def token_choice(
    token: str,
    choices: tuple[str, ...] = CHOICES,
) -> str | None:
    stripped = token.strip()
    return stripped if stripped in choices else None


def find_last_subsequence(
    sequence: list[int],
    subsequence: list[int],
) -> int | None:
    if not subsequence or len(subsequence) > len(sequence):
        return None

    for start in range(len(sequence) - len(subsequence), -1, -1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return start
    return None


def find_decision_position(
    token_ids: list[int],
    token_logprobs: list[dict[int, Any]],
    tokenizer: Any,
    prompt_style: str,
    thinking: bool,
    choices: tuple[str, ...] = CHOICES,
) -> int | None:
    if not token_ids or not token_logprobs:
        return None

    if thinking:
        think_end_ids = tokenizer.encode(
            "</think>",
            add_special_tokens=False,
        )
        think_end_start = find_last_subsequence(token_ids, think_end_ids)
        if think_end_start is None:
            return None

        response_start = think_end_start + len(think_end_ids)
        for position in range(response_start, len(token_ids)):
            token = tokenizer.decode([token_ids[position]])
            if token.strip():
                return position
        return None

    if prompt_style == "cot":
        # CoT has no structural delimiter, so use the final explicit choice.
        for position in range(len(token_ids) - 1, -1, -1):
            token_id = token_ids[position]
            logprob = token_logprobs[position].get(token_id)
            token = decoded_token(logprob, tokenizer, token_id)
            if token_choice(token, choices) is not None:
                return position
        return None

    # A direct prompt evaluates the distribution of the first generated token.
    return 0


def extract_choice_probabilities(
    position_logprobs: dict[int, Any],
    tokenizer: Any,
    choices: tuple[str, ...] = CHOICES,
) -> dict[str, Any]:
    log_probs: dict[str, float | None] = {choice: None for choice in choices}

    for token_id, logprob in position_logprobs.items():
        choice = token_choice(
            decoded_token(logprob, tokenizer, token_id),
            choices,
        )
        if choice is None:
            continue

        value = float(logprob.logprob)
        current = log_probs[choice]
        # Multiple token forms such as "是" and " 是" may represent one choice.
        log_probs[choice] = (
            value if current is None else float(np.logaddexp(current, value))
        )

    raw_probs = {
        choice: math.exp(value) if value is not None else None
        for choice, value in log_probs.items()
    }
    normalized_probs: dict[str, float | None] = {
        choice: None for choice in choices
    }
    if all(value is not None for value in log_probs.values()):
        denominator = sum(value for value in raw_probs.values() if value is not None)
        if denominator > 0:
            normalized_probs = {
                choice: raw_probs[choice] / denominator
                for choice in choices
            }

    return {
        "raw": raw_probs,
        "normalized": normalized_probs,
    }


def parse_answer(
    response: str,
    choices: tuple[str, ...] = CHOICES,
) -> str | None:
    choice_pattern = "|".join(
        re.escape(choice)
        for choice in sorted(choices, key=len, reverse=True)
    )
    matches = re.findall(
        rf"(?<![\w])(?:{choice_pattern})(?![\w])",
        response,
    )
    if matches:
        return matches[-1]

    stripped = response.strip()
    return stripped if stripped in choices else None


def process_outputs(
    outputs: list[Any],
    input_texts: list[str],
    tokenizer: Any,
    prompt_style: str,
    thinking: bool,
    choices: tuple[str, ...] = CHOICES,
) -> list[dict[str, Any]]:
    results = []
    for output, input_text in zip(outputs, input_texts):
        candidate = output.outputs[0]
        token_ids = list(candidate.token_ids)
        token_logprobs = candidate.logprobs or []
        decision_position = find_decision_position(
            token_ids,
            token_logprobs,
            tokenizer,
            prompt_style,
            thinking,
            choices,
        )

        probability = {
            "raw": {choice: None for choice in choices},
            "normalized": {choice: None for choice in choices},
        }
        if decision_position is not None:
            probability = extract_choice_probabilities(
                token_logprobs[decision_position],
                tokenizer,
                choices,
            )

        response = candidate.text.strip()
        results.append(
            {
                "input_text": input_text,
                "response": response,
                "parsed_answer": parse_answer(response, choices),
                "decision_token_position": decision_position,
                "choice_probability": probability,
            }
        )
    return results


def get_sampling_params(args: argparse.Namespace) -> Any:
    from vllm import SamplingParams

    is_long_output = args.prompt_style == "cot" or args.thinking
    if is_long_output:
        config = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": args.max_tokens or 8192,
        }
    else:
        config = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": args.max_tokens or 16,
        }

    config.update(
        {
            "logprobs": args.logprobs,
            "seed": args.seed,
        }
    )
    return SamplingParams(**config), config


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def run_identification(args: argparse.Namespace) -> Path:
    from vllm import LLM

    sentences, labels = load_dataset(args.dataset)
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
    suffix = "_thinking" if args.thinking else ""
    output_path = (
        Path(args.output_dir)
        / args.dataset
        / args.prompt_style
        / f"{model_name}{suffix}.json"
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
        "sampling_config": sampling_config,
        "results": [],
    }

    for start in tqdm(
        range(0, len(sentences), args.batch_size),
        desc="Identifying ambiguity",
    ):
        batch_sentences = sentences[start : start + args.batch_size]
        batch_labels = labels[start : start + args.batch_size]
        prompts = [
            prompt_template.format(sentence=sentence)
            for sentence in batch_sentences
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
        )

        for sentence, label, prediction in zip(
            batch_sentences,
            batch_labels,
            predictions,
        ):
            report["results"].append(
                {
                    "sentence": sentence,
                    "label": label,
                    "expected_answer": LABEL_TO_ANSWER[label],
                    **prediction,
                }
            )
        save_json(output_path, report)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ambiguity identification with vLLM."
    )
    parser.add_argument("--model_id", required=True, help="Hugging Face model ID")
    parser.add_argument("--dataset", default="CHA-Gen")
    parser.add_argument(
        "--prompt_style",
        choices=sorted(PROMPT_TEMPLATES),
        default="default",
    )
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--output_dir", default="results/identification")
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
    output_path = run_identification(args)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
