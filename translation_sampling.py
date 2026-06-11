#!/usr/bin/env python3
"""Sample Chinese-to-English translations with vLLM.

The output is a concatenated gzip stream. Each gzip member contains one
pickled record: first metadata, then one record per completed batch. This
keeps the output compact while allowing batch-level checkpointing.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
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


def load_source_lines(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            source = line.rstrip("\r\n")
            if source.strip():
                records.append(
                    {
                        "index": len(records),
                        "line_number": line_number,
                        "src": source,
                    }
                )
    return records


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
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "source_count": source_count,
        "sampling_params": sampling_config,
        "llm_config": {
            "tensor_parallel_size": args.tp_size,
            "max_num_seqs": args.max_num_seqs,
            "max_model_len": args.max_model_len,
            "enforce_eager": args.enforce_eager,
        },
    }


def config_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def state_path_for(output_path: Path) -> Path:
    return Path(f"{output_path}.state.json")


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8") as state_file:
        json.dump(data, state_file, ensure_ascii=False, indent=2)
        state_file.flush()
        os.fsync(state_file.fileno())
    os.replace(temporary, path)


def append_gzip_record(path: Path, record: dict[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as raw_file:
        with gzip.GzipFile(fileobj=raw_file, mode="wb") as compressed:
            pickle.dump(record, compressed, protocol=pickle.HIGHEST_PROTOCOL)
        raw_file.flush()
        os.fsync(raw_file.fileno())
        return raw_file.tell()


def initialize_or_resume(
    output_path: Path,
    config: dict[str, Any],
    overwrite: bool,
) -> int:
    state_path = state_path_for(output_path)
    if overwrite:
        output_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)

    output_exists = output_path.exists()
    state_exists = state_path.exists()
    if output_exists != state_exists:
        raise RuntimeError(
            "Output and checkpoint state are inconsistent. Pass --overwrite "
            f"to restart: {output_path}, {state_path}"
        )

    fingerprint = config_fingerprint(config)
    if not output_exists:
        metadata = {
            "type": "metadata",
            "created_at": time.time(),
            "hostname": socket.gethostname(),
            "config": config,
        }
        valid_size = append_gzip_record(output_path, metadata)
        atomic_write_json(
            state_path,
            {
                "format_version": OUTPUT_FORMAT_VERSION,
                "config_fingerprint": fingerprint,
                "completed_count": 0,
                "valid_size": valid_size,
            },
        )
        return 0

    with state_path.open("r", encoding="utf-8") as state_file:
        state = json.load(state_file)
    if state.get("format_version") != OUTPUT_FORMAT_VERSION:
        raise RuntimeError("Checkpoint format version does not match this script")
    if state.get("config_fingerprint") != fingerprint:
        raise RuntimeError(
            "Existing output was created with a different model, input, "
            "prompt, sampling configuration, or vLLM configuration. "
            "Pass --overwrite to restart."
        )

    valid_size = int(state["valid_size"])
    actual_size = output_path.stat().st_size
    if actual_size < valid_size:
        raise RuntimeError(
            f"Output is shorter than its checkpoint ({actual_size} < {valid_size})"
        )
    if actual_size != valid_size:
        with output_path.open("r+b") as output_file:
            output_file.truncate(valid_size)
            output_file.flush()
            os.fsync(output_file.fileno())
    return int(state["completed_count"])


def save_batch(
    output_path: Path,
    config: dict[str, Any],
    samples: list[dict[str, Any]],
    completed_count: int,
) -> None:
    batch_record = {
        "type": "batch",
        "start_index": samples[0]["index"],
        "end_index": samples[-1]["index"] + 1,
        "samples": samples,
    }
    valid_size = append_gzip_record(output_path, batch_record)
    atomic_write_json(
        state_path_for(output_path),
        {
            "format_version": OUTPUT_FORMAT_VERSION,
            "config_fingerprint": config_fingerprint(config),
            "completed_count": completed_count,
            "valid_size": valid_size,
        },
    )


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
    parser.add_argument("--input", required=True, help="One source sentence per line")
    parser.add_argument("--dataset", required=True, help="Dataset name for output")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument(
        "--logprobs",
        type=int,
        default=100,
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

    input_path = Path(args.input)
    sources = load_source_lines(input_path)
    if not sources:
        raise ValueError(f"No non-empty source lines found in {input_path}")

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
    output_path = args.output or default_output_path(args)
    completed_count = initialize_or_resume(
        output_path,
        config,
        args.overwrite,
    )
    if completed_count > len(sources):
        raise RuntimeError("Checkpoint contains more samples than the input")

    for start in range(completed_count, len(sources), args.batch_size):
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
        save_batch(output_path, config, samples, completed_count)
        print(f"Saved {completed_count}/{len(sources)} samples", flush=True)

    return output_path


def main() -> None:
    args = parse_args()
    output_path = run(args)
    print(f"Done. Results saved to {output_path}")


if __name__ == "__main__":
    main()
