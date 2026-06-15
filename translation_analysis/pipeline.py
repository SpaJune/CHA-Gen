#!/usr/bin/env python3
"""Run the translation analysis workflow through selected stages."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
STAGES = ("sample", "extract", "aggregate", "qe", "nli")


def sampling_output_path(model: str, dataset: str) -> Path:
    model_tag = model.rstrip("/").split("/")[-1].replace("-", "_").lower()
    return (
        ANALYSIS_DIR
        / "translation_outputs/prompt2"
        / dataset
        / f"{model_tag}.pkl.gz"
    )


def build_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    python = sys.executable
    commands: list[tuple[str, list[str]]] = []

    if "sample" in args.stages:
        if not args.translation_model:
            raise ValueError(
                "The sample stage requires at least one --translation-model"
            )
        for model in args.translation_model:
            command = [
                python,
                str(ANALYSIS_DIR / "translation_sampling.py"),
                "--model",
                model,
                "--model_type",
                args.model_type,
                "--dataset",
                args.dataset,
                "--n",
                str(args.n),
                "--max_tokens",
                str(args.max_tokens),
                "--temperature",
                str(args.temperature),
                "--top_p",
                str(args.top_p),
                "--top_k",
                str(args.top_k),
                "--logprobs",
                str(args.logprobs),
                "--seed",
                str(args.seed),
                "--batch_size",
                str(args.sampling_batch_size),
                "--tp_size",
                str(args.tp_size),
                "--max_num_seqs",
                str(args.max_num_seqs),
                "--max_model_len",
                str(args.max_model_len),
            ]
            if args.overwrite_sampling:
                command.append("--overwrite")
            commands.append((f"sample:{model}", command))

    if "extract" in args.stages:
        command = [
            python,
            str(ANALYSIS_DIR / "data_extraction.py"),
            "--input-dir",
            str(args.raw_dir),
            "--output-dir",
            str(args.extracted_dir),
        ]
        if args.overwrite_extracted:
            command.append("--overwrite")
        commands.append(("extract", command))

    if "aggregate" in args.stages:
        commands.append(
            (
                "aggregate",
                [
                    python,
                    str(ANALYSIS_DIR / "aggregate_translations.py"),
                    "--input-dir",
                    str(args.model_parquet_dir),
                    "--output",
                    str(args.unique_output),
                ],
            )
        )

    if "qe" in args.stages:
        command = [
            python,
            str(ANALYSIS_DIR / "scoring.py"),
            "--input",
            str(args.unique_output),
            "--model",
            args.qe_model,
            "--batch-size",
            str(args.qe_batch_size),
            "--chunk-size",
            str(args.qe_chunk_size),
            "--gpus",
            str(args.qe_gpus),
        ]
        if args.qe_devices:
            command.extend(["--devices", args.qe_devices])
        if args.allow_download:
            command.append("--allow-download")
        commands.append(("qe", command))

    if "nli" in args.stages:
        commands.append(
            (
                "nli",
                [
                    python,
                    str(ANALYSIS_DIR / "infer_nli.py"),
                    "--input",
                    str(args.unique_output),
                    "--model",
                    str(args.nli_model),
                    "--batch-size",
                    str(args.nli_batch_size),
                    "--pair-chunk-size",
                    str(args.pair_chunk_size),
                    "--max-length",
                    str(args.nli_max_length),
                    "--device",
                    args.nli_device,
                ],
            )
        )

    return commands


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selected translation sampling and analysis stages."
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGES,
        default=list(STAGES),
    )
    parser.add_argument(
        "--translation-model",
        action="append",
        help="Translation model ID; repeat for multiple models",
    )
    parser.add_argument("--dataset", default="CHA-Gen")
    parser.add_argument(
        "--model-type",
        choices=("auto", "base", "instruct"),
        default="auto",
    )
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--logprobs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-batch-size", type=int, default=128)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--overwrite-sampling", action="store_true")
    parser.add_argument("--overwrite-extracted", action="store_true")

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ANALYSIS_DIR / "translation_outputs",
    )
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=ANALYSIS_DIR / "extracted_outputs",
    )
    parser.add_argument(
        "--model-parquet-dir",
        type=Path,
        default=ANALYSIS_DIR / "extracted_outputs/prompt2/CHA-Gen",
    )
    parser.add_argument(
        "--unique-output",
        type=Path,
        default=ANALYSIS_DIR / "extracted_outputs/unique_translations.parquet",
    )

    parser.add_argument("--qe-model", default="Unbabel/wmt22-cometkiwi-da")
    parser.add_argument("--qe-batch-size", type=int, default=16)
    parser.add_argument("--qe-chunk-size", type=int, default=4096)
    parser.add_argument("--qe-gpus", type=int, default=1)
    parser.add_argument("--qe-devices")
    parser.add_argument("--allow-download", action="store_true")

    parser.add_argument(
        "--nli-model",
        type=Path,
        default=ANALYSIS_DIR / "paraphrase_checkpoints/checkpoint-2740",
    )
    parser.add_argument("--nli-batch-size", type=int, default=128)
    parser.add_argument("--pair-chunk-size", type=int, default=4096)
    parser.add_argument("--nli-max-length", type=int, default=384)
    parser.add_argument("--nli-device", default="auto")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    for name, command in build_commands(args):
        if name.startswith("sample:") and not args.overwrite_sampling:
            model = name.split(":", maxsplit=1)[1]
            output_path = sampling_output_path(model, args.dataset)
            if output_path.exists():
                print(f"SKIP {name}: {output_path} already exists", flush=True)
                continue

        print(f"\nRUN  {name}\n     {' '.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    try:
        run(parse_args())
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
