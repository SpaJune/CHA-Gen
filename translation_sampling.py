#!/usr/bin/env python3
"""Sample Chinese-to-English translations with vLLM.

The output is a concatenated gzip stream. Each gzip member contains one
pickled record: first metadata, then one record per completed batch. This
keeps the output compact while allowing batch-level writes.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import os
import pickle
import socket
import time
from pathlib import Path
from typing import Any, Iterable


PROMPT_VERSION = "prompt2-v1"
PROMPT_INSTRUCTION = "Translate this from Chinese into English:"
ASSISTANT_PREFIX = 'English: "'
OUTPUT_FORMAT_VERSION = 1


def infer_model_type(model_id: str, requested: str = "auto") -> str | None:
    """Infer base/instruct for known model families.

    None means that auto detection must be completed after loading the
    tokenizer.
    """
    if requested != "auto":
        return requested

    name = model_id.rstrip("/").split("/")[-1].lower()
    if "qwen3" in name:
        return "base" if "base" in name else "instruct"
    if "qwen2.5" in name:
        return "instruct" if "instruct" in name else "base"
    return None


def resolve_model_type(
    model_id: str,
    requested: str = "auto",
    tokenizer: Any | None = None,
) -> str:
    model_type = infer_model_type(model_id, requested)
    if model_type is not None:
        return model_type

    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return "instruct"
    raise ValueError(
        f"Cannot infer whether {model_id!r} is a base or instruct model. "
        "Pass --model_type base or --model_type instruct."
    )


def semantic_prompt(src: str) -> str:
    return f'{PROMPT_INSTRUCTION}\nChinese: "{src}"'


def build_prompt(
    src: str,
    model_id: str,
    model_type: str,
    tokenizer: Any | None = None,
) -> str:
    content = semantic_prompt(src)
    if model_type == "base":
        return f"{content}\n{ASSISTANT_PREFIX}"
    if tokenizer is None:
        raise ValueError("An instruct model requires a tokenizer")

    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if "qwen3" in model_id.lower():
        template_kwargs["enable_thinking"] = False
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        **template_kwargs,
    )
    return f"{rendered}{ASSISTANT_PREFIX}"


def safe_decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        try:
            token = tokenizer.convert_ids_to_tokens(int(token_id))
            if isinstance(token, list):
                return str(token[0])
            return str(token)
        except Exception:
            return str(token_id)


def _candidate_token_id(key: Any, value: Any) -> int | None:
    for candidate in (
        key,
        getattr(value, "token_id", None),
        getattr(value, "logprob_token_id", None),
    ):
        if candidate is None:
            continue
        try:
            return int(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _serialize_logprob(
    token_id: int | None,
    value: Any,
    tokenizer: Any,
) -> dict[str, Any] | None:
    try:
        logprob = float(getattr(value, "logprob", value))
    except (TypeError, ValueError):
        return None

    decoded = getattr(value, "decoded_token", None)
    if decoded is None:
        decoded = (
            safe_decode_token(tokenizer, token_id)
            if token_id is not None
            else None
        )
    rank = getattr(value, "rank", None)
    return {
        "token_id": token_id,
        "token": decoded,
        "logprob": logprob,
        "rank": int(rank) if rank is not None else None,
    }


def extract_token_level_logprobs(
    completion: Any,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    token_ids = [int(token_id) for token_id in completion.token_ids]
    positions = getattr(completion, "logprobs", None)
    result: list[dict[str, Any]] = []

    for position, sampled_id in enumerate(token_ids):
        top: list[dict[str, Any]] = []
        sampled_logprob = None
        entry = positions[position] if positions is not None and position < len(positions) else None

        if isinstance(entry, dict):
            values: Iterable[tuple[Any, Any]] = entry.items()
        elif isinstance(entry, (list, tuple)):
            values = ((None, value) for value in entry)
        else:
            values = ()

        for key, value in values:
            candidate_id = _candidate_token_id(key, value)
            serialized = _serialize_logprob(candidate_id, value, tokenizer)
            if serialized is None:
                continue
            top.append(serialized)
            if candidate_id == sampled_id:
                sampled_logprob = serialized["logprob"]

        result.append(
            {
                "token_id": sampled_id,
                "token": safe_decode_token(tokenizer, sampled_id),
                "sampled_logprob": sampled_logprob,
                "top": top or None,
            }
        )
    return result


def serialize_completion(completion: Any, tokenizer: Any) -> dict[str, Any]:
    return {
        "text": completion.text,
        "cumulative_logprob": getattr(completion, "cumulative_logprob", None),
        "token_ids": [int(token_id) for token_id in completion.token_ids],
        "tokens": extract_token_level_logprobs(completion, tokenizer),
        "finish_reason": getattr(completion, "finish_reason", None),
        "stop_reason": getattr(completion, "stop_reason", None),
    }


def dataset_path(name: str) -> Path:
    if name == "CHA-Gen":
        return Path("dataset/CHA-Gen/corpus.csv")
    raise ValueError(f"Unsupported dataset: {name}")


def load_dataset(name: str) -> tuple[Path, list[dict[str, Any]]]:
    path = dataset_path(name)
    records = []
    with path.open("r", encoding="utf-8", newline="") as dataset_file:
        reader = csv.DictReader(dataset_file)
        required_columns = {"sent", "ambiguity"}
        missing_columns = required_columns - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                f"Dataset is missing columns: {sorted(missing_columns)}"
            )

        for row_number, row in enumerate(reader, start=2):
            source = row["sent"]
            if not source or not source.strip():
                raise ValueError(f"Empty sentence at {path}:{row_number}")
            try:
                ambiguity = int(row["ambiguity"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid ambiguity label at {path}:{row_number}"
                ) from exc
            if ambiguity not in (0, 1):
                raise ValueError(
                    f"Unexpected ambiguity label at {path}:{row_number}: "
                    f"{ambiguity}"
                )
            records.append(
                {
                    "index": len(records),
                    "row_number": row_number,
                    "src": source,
                    "ambiguity": ambiguity,
                }
            )
    return path, records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_config(
    args: argparse.Namespace,
    model_type: str,
    input_path: Path,
    source_count: int,
    sampling_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": OUTPUT_FORMAT_VERSION,
        "model": args.model,
        "model_type": model_type,
        "dataset": args.dataset,
        "prompt_version": PROMPT_VERSION,
        "prompt": {
            "semantic_template": (
                f'{PROMPT_INSTRUCTION}\nChinese: "{{src}}"'
            ),
            "assistant_prefix": ASSISTANT_PREFIX,
            "instruct_uses_chat_template": True,
            "qwen3_thinking": False,
            "stop": None,
        },
        "dataset_path": str(input_path.resolve()),
        "dataset_sha256": sha256_file(input_path),
        "source_count": source_count,
        "sampling_params": sampling_config,
        "llm_config": {
            "tensor_parallel_size": args.tp_size,
            "max_num_seqs": args.max_num_seqs,
            "max_model_len": args.max_model_len,
            "enforce_eager": args.enforce_eager,
        },
    }


def legacy_state_path_for(output_path: Path) -> Path:
    return Path(f"{output_path}.state.json")


def write_gzip_record(
    path: Path,
    record: dict[str, Any],
    mode: str = "ab",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode) as raw_file:
        with gzip.GzipFile(fileobj=raw_file, mode="wb") as compressed:
            pickle.dump(record, compressed, protocol=pickle.HIGHEST_PROTOCOL)
        raw_file.flush()
        os.fsync(raw_file.fileno())


def existing_output_paths(output_path: Path) -> list[Path]:
    return [
        path
        for path in (output_path, legacy_state_path_for(output_path))
        if path.exists()
    ]


def ensure_output_available(output_path: Path, overwrite: bool) -> None:
    existing = existing_output_paths(output_path)
    if existing and not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output already exists: {paths}; pass --overwrite to replace it"
        )


def initialize_output(
    output_path: Path,
    config: dict[str, Any],
    overwrite: bool,
) -> None:
    ensure_output_available(output_path, overwrite)
    if overwrite:
        output_path.unlink(missing_ok=True)
        legacy_state_path_for(output_path).unlink(missing_ok=True)

    write_gzip_record(
        output_path,
        {
            "type": "metadata",
            "created_at": time.time(),
            "hostname": socket.gethostname(),
            "config": config,
        },
        mode="xb",
    )


def save_batch(
    output_path: Path,
    samples: list[dict[str, Any]],
) -> None:
    batch_record = {
        "type": "batch",
        "start_index": samples[0]["index"],
        "end_index": samples[-1]["index"] + 1,
        "samples": samples,
    }
    write_gzip_record(output_path, batch_record)


def default_output_path(args: argparse.Namespace) -> Path:
    model_tag = args.model.rstrip("/").split("/")[-1].replace("-", "_").lower()
    return (
        Path("translation_outputs")
        / PROMPT_VERSION.split("-", maxsplit=1)[0]
        / args.dataset
        / f"{model_tag}.pkl.gz"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample Chinese-to-English translations with vLLM."
    )
    parser.add_argument("--model", required=True, help="Model ID or local path")
    parser.add_argument(
        "--model_type",
        choices=("auto", "base", "instruct"),
        default="auto",
    )
    parser.add_argument(
        "--dataset",
        default="CHA-Gen",
        help="Dataset to load (currently supported: CHA-Gen)",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument(
        "--logprobs",
        type=int,
        default=20,
        help="Number of top token logprobs retained at each generated position",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--enforce_eager", action="store_true")
    args = parser.parse_args()

    for name in ("n", "max_tokens", "batch_size", "tp_size", "max_num_seqs"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.logprobs == 0 or args.logprobs < -1:
        parser.error("--logprobs must be positive or -1")
    return args


def run(args: argparse.Namespace) -> Path:
    from vllm import LLM, SamplingParams

    input_path, sources = load_dataset(args.dataset)
    if not sources:
        raise ValueError(f"No source sentences found in {input_path}")

    output_path = args.output or default_output_path(args)
    ensure_output_available(output_path, args.overwrite)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )
    tokenizer = llm.get_tokenizer()
    model_type = resolve_model_type(
        args.model,
        args.model_type,
        tokenizer,
    )

    sampling_config = {
        "n": args.n,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k if args.top_k > 0 else -1,
        "logprobs": args.logprobs,
        "seed": args.seed,
    }
    sampling_params = SamplingParams(**sampling_config)
    config = experiment_config(
        args,
        model_type,
        input_path,
        len(sources),
        sampling_config,
    )
    initialize_output(
        output_path,
        config,
        args.overwrite,
    )

    for start in range(0, len(sources), args.batch_size):
        batch = sources[start : start + args.batch_size]
        prompts = [
            build_prompt(
                item["src"],
                args.model,
                model_type,
                tokenizer,
            )
            for item in batch
        ]
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        if len(outputs) != len(batch):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(batch)} prompts"
            )

        samples = []
        for item, prompt, request_output in zip(batch, prompts, outputs):
            samples.append(
                {
                    **item,
                    "prompt": prompt,
                    "candidates": [
                        serialize_completion(completion, tokenizer)
                        for completion in request_output.outputs
                    ],
                }
            )
        completed_count = start + len(batch)
        save_batch(output_path, samples)
        print(f"Saved {completed_count}/{len(sources)} samples", flush=True)

    return output_path


def main() -> None:
    args = parse_args()
    output_path = run(args)
    print(f"Done. Results saved to {output_path}")


if __name__ == "__main__":
    main()
