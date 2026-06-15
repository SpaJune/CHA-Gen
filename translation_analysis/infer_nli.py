#!/usr/bin/env python3
"""Score translation pairs with an incrementally cached equivalence model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = ANALYSIS_DIR / "extracted_outputs/unique_translations.parquet"
DEFAULT_MODEL = ANALYSIS_DIR / "paraphrase_checkpoints/checkpoint-2740"
DEFAULT_CACHE_ROOT = ANALYSIS_DIR / "cache/nli"
DEFAULT_OUTPUT_ROOT = ANALYSIS_DIR / "scored_outputs/nli"
REQUIRED_COLUMNS = (
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
        ("sentence1", pa.string()),
        ("sentence2", pa.string()),
        ("prediction", pa.string()),
        ("equivalent_prob", pa.float64()),
        ("neutral_prob", pa.float64()),
        ("contradiction_prob", pa.float64()),
        ("is_equivalent", pa.bool_()),
        ("nli_model", pa.string()),
        ("nli_revision", pa.string()),
    ]
)
LABEL_ALIASES = {
    "entailment": "equivalent",
    "equivalent": "equivalent",
    "neutral": "neutral",
    "not equivalent": "neutral",
    "not_equivalent": "neutral",
    "contradiction": "contradiction",
}
DISPLAY_LABELS = {
    "equivalent": "Equivalent",
    "neutral": "Neutral",
    "contradiction": "Contradiction",
}


def model_slug(model_path: str | Path) -> str:
    name = Path(model_path).expanduser().resolve().name.lower()
    slug = "".join(character if character.isalnum() else "_" for character in name)
    slug = "_".join(part for part in slug.split("_") if part)
    if not slug:
        raise ValueError(f"Cannot derive a directory name from model path {model_path!r}")
    return slug


def checkpoint_revision(model_path: str | Path) -> str:
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Model checkpoint does not exist: {path}")

    tracked_names = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "spm.model",
        "model.safetensors",
        "pytorch_model.bin",
    )
    records = []
    for name in tracked_names:
        file_path = path / name
        if file_path.is_file():
            stat = file_path.stat()
            records.append((name, stat.st_size, stat.st_mtime_ns))
    if not records:
        raise ValueError(f"No model or tokenizer files found under {path}")

    payload = json.dumps(records, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def canonical_pair(sentence1: str, sentence2: str) -> tuple[str, str]:
    if sentence1 == sentence2:
        raise ValueError("A sentence cannot be paired with itself")
    return (
        (sentence1, sentence2)
        if sentence1 < sentence2
        else (sentence2, sentence1)
    )


def cache_key(
    model_id: str,
    revision: str,
    sentence1: str,
    sentence2: str,
) -> str:
    first, second = canonical_pair(sentence1, sentence2)
    payload = json.dumps(
        [model_id, revision, first, second],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_input(frame: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    invalid = frame.index[
        frame["source_text"].isna()
        | frame["processed_text"].isna()
        | frame["source_text"].astype(str).str.strip().eq("")
        | frame["processed_text"].astype(str).str.strip().eq("")
    ].tolist()
    if invalid:
        preview = ", ".join(str(index) for index in invalid[:20])
        raise ValueError(f"Empty source or translation text at input rows: {preview}")

    if frame.duplicated(["source_index", "processed_text"]).any():
        raise ValueError(
            "Input contains duplicate source_index/processed_text rows; "
            "use the unique translation table"
        )

    metadata_counts = frame.groupby("source_index", sort=False)[
        ["source_text", "ambiguity"]
    ].nunique(dropna=False)
    if metadata_counts.gt(1).any(axis=None):
        raise ValueError("A source_index has inconsistent source metadata")


def iter_source_groups(
    frame: pd.DataFrame,
) -> Iterator[tuple[int, str, int, list[str]]]:
    for source_index, group in frame.groupby("source_index", sort=True):
        yield (
            int(source_index),
            str(group["source_text"].iloc[0]),
            int(group["ambiguity"].iloc[0]),
            sorted(str(text) for text in group["processed_text"]),
        )


def iter_pairs(sentences: Sequence[str]) -> Iterator[tuple[str, str]]:
    for first_index, sentence1 in enumerate(sentences):
        for sentence2 in sentences[first_index + 1 :]:
            yield canonical_pair(sentence1, sentence2)


def chunked(
    values: Iterator[tuple[str, str]],
    chunk_size: int,
) -> Iterator[list[tuple[str, str]]]:
    chunk = []
    for value in values:
        chunk.append(value)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def open_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            cache_key TEXT PRIMARY KEY,
            sentence1 TEXT NOT NULL,
            sentence2 TEXT NOT NULL,
            equivalent_prob REAL NOT NULL,
            neutral_prob REAL NOT NULL,
            contradiction_prob REAL NOT NULL,
            prediction TEXT NOT NULL,
            is_equivalent INTEGER NOT NULL,
            nli_model TEXT NOT NULL,
            nli_revision TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS predictions_pair "
        "ON predictions(sentence1, sentence2)"
    )
    return connection


def fetch_cached(
    connection: sqlite3.Connection,
    keys: Sequence[str],
) -> dict[str, sqlite3.Row]:
    if not keys:
        return {}
    previous_factory = connection.row_factory
    connection.row_factory = sqlite3.Row
    rows: dict[str, sqlite3.Row] = {}
    try:
        for start in range(0, len(keys), 900):
            key_chunk = keys[start : start + 900]
            placeholders = ",".join("?" for _ in key_chunk)
            query = f"SELECT * FROM predictions WHERE cache_key IN ({placeholders})"
            rows.update(
                (row["cache_key"], row)
                for row in connection.execute(query, key_chunk)
            )
    finally:
        connection.row_factory = previous_factory
    return rows


def normalize_label(label: str) -> str:
    normalized = str(label).strip().lower()
    try:
        return LABEL_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported model label: {label!r}") from exc


def label_indices(model: Any) -> dict[str, int]:
    result = {
        normalize_label(label): int(index)
        for index, label in model.config.id2label.items()
    }
    expected = {"equivalent", "neutral", "contradiction"}
    if set(result) != expected:
        raise ValueError(
            f"Model labels must map to {sorted(expected)}, got {sorted(result)}"
        )
    return result


def predict_probabilities(
    model: Any,
    tokenizer: Any,
    pairs: Sequence[tuple[str, str]],
    batch_size: int,
    max_length: int,
    device: str,
) -> list[tuple[float, float, float]]:
    import torch

    indices = label_indices(model)
    results = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        inputs = tokenizer(
            [pair[0] for pair in batch],
            [pair[1] for pair in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            probabilities = torch.softmax(model(**inputs).logits, dim=-1).cpu()
        for probability in probabilities:
            results.append(
                (
                    float(probability[indices["equivalent"]]),
                    float(probability[indices["neutral"]]),
                    float(probability[indices["contradiction"]]),
                )
            )
    return results


def validate_probabilities(
    probabilities: Sequence[Sequence[float]],
    expected_count: int,
) -> None:
    if len(probabilities) != expected_count:
        raise RuntimeError(
            f"Scorer returned {len(probabilities)} rows for {expected_count} pairs"
        )
    for values in probabilities:
        if len(values) != 3:
            raise RuntimeError("Scorer must return three probabilities per pair")
        if not all(math.isfinite(float(value)) for value in values):
            raise RuntimeError("Scorer returned a non-finite probability")
        if any(float(value) < 0.0 or float(value) > 1.0 for value in values):
            raise RuntimeError("Scorer returned a probability outside [0, 1]")
        if not math.isclose(sum(map(float, values)), 1.0, abs_tol=1e-4):
            raise RuntimeError("Scorer probabilities do not sum to one")


def cache_missing_pairs(
    connection: sqlite3.Connection,
    pairs: Sequence[tuple[str, str]],
    model_id: str,
    revision: str,
    score_pairs: Callable[
        [Sequence[tuple[str, str]]],
        Sequence[Sequence[float]],
    ],
) -> tuple[list[str], int]:
    keys = [
        cache_key(model_id, revision, sentence1, sentence2)
        for sentence1, sentence2 in pairs
    ]
    cached = fetch_cached(connection, keys)
    missing_indices = [
        index for index, key in enumerate(keys) if key not in cached
    ]
    if not missing_indices:
        return keys, 0

    missing_pairs = [pairs[index] for index in missing_indices]
    probabilities = score_pairs(missing_pairs)
    validate_probabilities(probabilities, len(missing_pairs))

    rows = []
    for index, values in zip(missing_indices, probabilities):
        equivalent, neutral, contradiction = map(float, values)
        canonical_label = max(
            (
                ("equivalent", equivalent),
                ("neutral", neutral),
                ("contradiction", contradiction),
            ),
            key=lambda item: item[1],
        )[0]
        sentence1, sentence2 = pairs[index]
        rows.append(
            (
                keys[index],
                sentence1,
                sentence2,
                equivalent,
                neutral,
                contradiction,
                DISPLAY_LABELS[canonical_label],
                int(canonical_label == "equivalent"),
                model_id,
                revision,
            )
        )

    with connection:
        connection.executemany(
            """
            INSERT OR IGNORE INTO predictions (
                cache_key, sentence1, sentence2,
                equivalent_prob, neutral_prob, contradiction_prob,
                prediction, is_equivalent, nli_model, nli_revision
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return keys, len(rows)


def output_rows(
    source_index: int,
    source_text: str,
    ambiguity: int,
    pairs: Sequence[tuple[str, str]],
    keys: Sequence[str],
    cached: dict[str, sqlite3.Row],
) -> list[dict[str, Any]]:
    rows = []
    for pair, key in zip(pairs, keys):
        result = cached[key]
        rows.append(
            {
                "source_index": source_index,
                "source_text": source_text,
                "ambiguity": ambiguity,
                "sentence1": pair[0],
                "sentence2": pair[1],
                "prediction": result["prediction"],
                "equivalent_prob": result["equivalent_prob"],
                "neutral_prob": result["neutral_prob"],
                "contradiction_prob": result["contradiction_prob"],
                "is_equivalent": bool(result["is_equivalent"]),
                "nli_model": result["nli_model"],
                "nli_revision": result["nli_revision"],
            }
        )
    return rows


def score_dataframe(
    frame: pd.DataFrame,
    connection: sqlite3.Connection,
    output_path: Path,
    model_id: str,
    revision: str,
    pair_chunk_size: int,
    score_pairs: Callable[
        [Sequence[tuple[str, str]]],
        Sequence[Sequence[float]],
    ],
) -> tuple[int, int]:
    validate_input(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.unlink(missing_ok=True)

    writer = pq.ParquetWriter(temporary_path, OUTPUT_SCHEMA, compression="zstd")
    total_pairs = 0
    scored_pairs = 0
    try:
        for source_index, source_text, ambiguity, sentences in iter_source_groups(
            frame
        ):
            for pairs in chunked(iter_pairs(sentences), pair_chunk_size):
                keys, newly_scored = cache_missing_pairs(
                    connection,
                    pairs,
                    model_id,
                    revision,
                    score_pairs,
                )
                cached = fetch_cached(connection, keys)
                if len(cached) != len(set(keys)):
                    raise RuntimeError("Some sentence pairs remain uncached")
                rows = output_rows(
                    source_index,
                    source_text,
                    ambiguity,
                    pairs,
                    keys,
                    cached,
                )
                writer.write_table(pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA))
                total_pairs += len(rows)
                scored_pairs += newly_scored
            print(
                f"Processed source {source_index}: {len(sentences):,} translations; "
                f"{total_pairs:,} total pairs, {scored_pairs:,} newly scored",
                flush=True,
            )
    except Exception:
        writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        writer.close()
        temporary_path.replace(output_path)
    return total_pairs, scored_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score all within-source translation pairs with an NLI model."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pair-chunk-size", type=int, default=4096)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device such as auto, cpu, cuda, or cuda:1.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.pair_chunk_size <= 0:
        parser.error("--pair-chunk-size must be positive")
    if args.max_length <= 0:
        parser.error("--max-length must be positive")
    return args


def run(args: argparse.Namespace) -> Path:
    import torch

    input_path = args.input.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input Parquet does not exist: {input_path}")

    model_id = str(model_path)
    revision = checkpoint_revision(model_path)
    slug = model_slug(model_path)
    cache_path = DEFAULT_CACHE_ROOT / slug / "predictions.sqlite"
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else DEFAULT_OUTPUT_ROOT / slug / "translation_pairs.parquet"
    )
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )

    frame = pd.read_parquet(input_path, columns=list(REQUIRED_COLUMNS))
    validate_input(frame)
    pair_count = sum(
        len(sentences) * (len(sentences) - 1) // 2
        for _, _, _, sentences in iter_source_groups(frame)
    )
    print(
        f"Model: {model_id}\n"
        f"Revision: {revision}\n"
        f"Input translations: {len(frame):,}\n"
        f"Translation pairs: {pair_count:,}\n"
        f"Device: {device}\n"
        f"Cache: {cache_path}\n"
        f"Output: {output_path}",
        flush=True,
    )

    model = None
    tokenizer = None

    def score_pairs(
        pairs: Sequence[tuple[str, str]],
    ) -> Sequence[Sequence[float]]:
        nonlocal model, tokenizer
        if model is None or tokenizer is None:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                local_files_only=True,
            )
            model.to(device)
            model.eval()
            label_indices(model)
        return predict_probabilities(
            model,
            tokenizer,
            pairs,
            args.batch_size,
            args.max_length,
            device,
        )

    connection = open_cache(cache_path)
    try:
        total_pairs, scored_pairs = score_dataframe(
            frame,
            connection,
            output_path,
            model_id,
            revision,
            args.pair_chunk_size,
            score_pairs,
        )
    finally:
        connection.close()

    print(
        f"Saved {total_pairs:,} pairs to {output_path}; "
        f"{scored_pairs:,} required new inference",
        flush=True,
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
