#!/usr/bin/env python3
"""Extract lightweight translation records from sampling output to Parquet."""

from __future__ import annotations

import argparse
import gzip
import pickle
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "translation_outputs"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "extracted_outputs"
INPUT_SUFFIX = ".pkl.gz"

SCHEMA = pa.schema(
    [
        ("source_index", pa.int64()),
        ("row_number", pa.int64()),
        ("source_text", pa.string()),
        ("ambiguity", pa.int8()),
        ("model", pa.string()),
        ("model_type", pa.string()),
        ("candidate_index", pa.int32()),
        ("raw_text", pa.string()),
        ("processed_text", pa.string()),
        ("finish_reason", pa.string()),
    ]
)


def parse_translation(text: Any) -> str:
    """Apply the original lightweight notebook cleanup."""
    if not isinstance(text, str):
        return ""

    parsed = text.strip()
    if parsed.endswith('"'):
        parsed = parsed[:-1].strip()
    if "\n" in parsed:
        parsed = parsed.split("\n", maxsplit=1)[0].strip()
        if parsed.endswith('"'):
            parsed = parsed[:-1].strip()
    if "Note: " in parsed:
        parsed = parsed.split("Note: ", maxsplit=1)[0].strip()
    return parsed


def iter_pickle_records(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rb") as input_file:
        while True:
            try:
                record = pickle.load(input_file)
            except EOFError:
                return
            if not isinstance(record, dict):
                raise ValueError(f"Expected a dictionary record in {path}")
            yield record


def output_path_for(
    input_path: Path,
    input_dir: Path,
    output_dir: Path,
) -> Path:
    relative_path = input_path.relative_to(input_dir)
    return output_dir / relative_path.with_name(
        relative_path.name[: -len(INPUT_SUFFIX)] + ".parquet"
    )


def rows_from_batch(
    record: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    model = str(config.get("model", ""))
    model_type = str(config.get("model_type", ""))

    for sample in record.get("samples", []):
        for candidate_index, candidate in enumerate(sample.get("candidates", [])):
            raw_text = candidate.get("text", "")
            finish_reason = candidate.get("finish_reason")
            rows.append(
                {
                    "source_index": sample.get("index"),
                    "row_number": sample.get("row_number"),
                    "source_text": sample.get("src", ""),
                    "ambiguity": sample.get("ambiguity"),
                    "model": model,
                    "model_type": model_type,
                    "candidate_index": candidate_index,
                    "raw_text": (
                        raw_text if isinstance(raw_text, str) else str(raw_text)
                    ),
                    "processed_text": parse_translation(raw_text),
                    "finish_reason": (
                        None if finish_reason is None else str(finish_reason)
                    ),
                }
            )
    return rows


def extract_file(
    input_path: Path,
    output_path: Path,
    overwrite: bool = False,
) -> int | None:
    """Extract one file and return its row count, or None when skipped."""
    if output_path.exists() and not overwrite:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.unlink(missing_ok=True)

    writer: pq.ParquetWriter | None = None
    config: dict[str, Any] | None = None
    row_count = 0
    try:
        for record in iter_pickle_records(input_path):
            record_type = record.get("type")
            if record_type == "metadata":
                config = record.get("config")
                if not isinstance(config, dict):
                    raise ValueError(f"Invalid metadata config in {input_path}")
            elif record_type == "batch":
                if config is None:
                    raise ValueError(
                        f"Batch encountered before metadata in {input_path}"
                    )
                rows = rows_from_batch(record, config)
                if not rows:
                    continue
                table = pa.Table.from_pylist(rows, schema=SCHEMA)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary_path,
                        SCHEMA,
                        compression="zstd",
                    )
                writer.write_table(table)
                row_count += len(rows)
            else:
                raise ValueError(
                    f"Unsupported record type {record_type!r} in {input_path}"
                )

        if config is None:
            raise ValueError(f"No metadata record found in {input_path}")
        if writer is None:
            writer = pq.ParquetWriter(
                temporary_path,
                SCHEMA,
                compression="zstd",
            )
    except Exception:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        writer.close()
        temporary_path.replace(output_path)
        return row_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text-level translation data to mirrored Parquet files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input tree (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output tree (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing Parquet files instead of skipping them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    input_paths = sorted(input_dir.rglob(f"*{INPUT_SUFFIX}"))
    if not input_paths:
        print(f"No {INPUT_SUFFIX} files found under {input_dir}", file=sys.stderr)
        return 2

    succeeded: list[tuple[Path, int]] = []
    skipped: list[Path] = []
    failed: list[tuple[Path, str]] = []

    for input_path in input_paths:
        output_path = output_path_for(input_path, input_dir, output_dir)
        try:
            row_count = extract_file(input_path, output_path, args.overwrite)
        except Exception as exc:
            failed.append((input_path, str(exc)))
            print(f"FAILED  {input_path}: {exc}", file=sys.stderr)
        else:
            if row_count is None:
                skipped.append(output_path)
                print(f"SKIPPED {output_path}")
            else:
                succeeded.append((output_path, row_count))
                print(f"WROTE   {output_path} ({row_count:,} rows)")

    print(
        f"\nSummary: {len(succeeded)} written, {len(skipped)} skipped, "
        f"{len(failed)} failed"
    )
    if failed:
        print("Failed files:", file=sys.stderr)
        for path, error in failed:
            print(f"  {path}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
