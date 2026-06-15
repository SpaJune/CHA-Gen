#!/usr/bin/env python3
"""Aggregate extracted model outputs into one unique translation table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ANALYSIS_DIR / "extracted_outputs/prompt2/CHA-Gen"
DEFAULT_OUTPUT = ANALYSIS_DIR / "extracted_outputs/unique_translations.parquet"
INPUT_COLUMNS = (
    "source_index",
    "source_text",
    "ambiguity",
    "processed_text",
)
OUTPUT_SCHEMA = pa.schema(
    [
        ("source_index", pa.int64()),
        ("source_text", pa.string()),
        ("ambiguity", pa.int8()),
        ("processed_text", pa.string()),
    ]
)


def aggregate_translations(input_paths: list[Path]) -> tuple[pd.DataFrame, int]:
    if not input_paths:
        raise ValueError("No extracted Parquet files were provided")

    frames = [
        pd.read_parquet(path, columns=list(INPUT_COLUMNS))
        for path in input_paths
    ]
    translations = pd.concat(frames, ignore_index=True)

    metadata_counts = translations.groupby("source_index", sort=False)[
        ["source_text", "ambiguity"]
    ].nunique(dropna=False)
    if metadata_counts.gt(1).any(axis=None):
        raise ValueError("A source_index has inconsistent source metadata")

    empty_mask = (
        translations["processed_text"].isna()
        | translations["processed_text"].astype(str).str.strip().eq("")
    )
    unique_translations = (
        translations.loc[~empty_mask, list(INPUT_COLUMNS)]
        .drop_duplicates(["source_index", "processed_text"])
        .sort_values(["source_index", "processed_text"], ignore_index=True)
    )
    return unique_translations, int(empty_mask.sum())


def write_parquet_atomic(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        table = pa.Table.from_pandas(
            frame,
            schema=OUTPUT_SCHEMA,
            preserve_index=False,
            safe=True,
        )
        pq.write_table(table, temporary_path, compression="zstd")
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate extracted translations across model outputs."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    input_dir = args.input_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    input_paths = sorted(input_dir.glob("*.parquet"))
    if not input_paths:
        raise FileNotFoundError(f"No Parquet files found under {input_dir}")

    frame, empty_count = aggregate_translations(input_paths)
    write_parquet_atomic(frame, output_path)
    print(
        f"Loaded {len(input_paths)} model files; excluded {empty_count:,} empty "
        f"samples; saved {len(frame):,} unique translations to {output_path}"
    )
    return output_path


def main() -> int:
    try:
        run(parse_args())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
