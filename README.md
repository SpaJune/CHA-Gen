# CHA-Gen

## Output Data Format

`translation_analysis/translation_sampling.py` generates translation sampling
data in **concatenated gzip stream** format (`.pkl.gz`), where each gzip member
contains one pickle-serialized record.

### Default Output Path

```
translation_analysis/translation_outputs/prompt2/CHA-Gen/{model_tag}.pkl.gz
```

### Record Types

The file contains two types of records, distinguished by the `"type"` field:

#### 1. Metadata Record (First Record)

Contains the full experiment configuration:

```json
{
  "type": "metadata",
  "created_at": <float>,
  "hostname": <str>,
  "config": {
    "format_version": 1,
    "model": <str>,
    "model_type": "base" | "instruct",
    "dataset": "CHA-Gen",
    "prompt_version": "prompt2-v1",
    "prompt": {
      "semantic_template": "Translate this from Chinese into English:\nChinese: \"{{src}}\"",
      "assistant_prefix": "English: \"",
      "instruct_uses_chat_template": true,
      "qwen3_thinking": false,
      "stop": null
    },
    "dataset_path": <str>,
    "dataset_sha256": <str>,
    "source_count": <int>,
    "sampling_params": {
      "n": <int>,
      "max_tokens": <int>,
      "temperature": <float>,
      "top_p": <float>,
      "top_k": <int>,
      "logprobs": <int>,
      "seed": <int>
    },
    "llm_config": {
      "tensor_parallel_size": <int>,
      "max_num_seqs": <int>,
      "max_model_len": <int>,
      "enforce_eager": <bool>
    }
  }
}
```

#### 2. Batch Record (One per Batch)

```json
{
  "type": "batch",
  "start_index": <int>,
  "end_index": <int>,
  "samples": [
    {
      "index": <int>,
      "row_number": <int>,
      "src": <str>,
      "ambiguity": 0 | 1,
      "prompt": <str>,
      "candidates": [
        {
          "text": <str>,
          "cumulative_logprob": <float>,
          "token_ids": [<int>, ...],
          "tokens": [
            {
              "token_id": <int>,
              "token": <str>,
              "sampled_logprob": <float>,
              "top": [
                {
                  "token_id": <int>,
                  "token": <str>,
                  "logprob": <float>,
                  "rank": <int>
                }
              ]
            }
          ],
          "finish_reason": <str>,
          "stop_reason": <str>
        }
      ]
    }
  ]
}
```

### Field Description

| Field | Type | Description |
|-------|------|-------------|
| `index` | int | Sample index in the dataset |
| `row_number` | int | Row number in the original CSV |
| `src` | str | Chinese source sentence |
| `ambiguity` | int | Ambiguity label (0=unambiguous, 1=ambiguous) |
| `prompt` | str | Full prompt string |
| `candidates` | list | List of translation candidates (length = `--n`) |
| `text` | str | Generated translation text |
| `cumulative_logprob` | float | Cumulative log probability |
| `token_ids` | list[int] | Sequence of generated token IDs |
| `tokens` | list[dict] | Per-token detailed information |
| `sampled_logprob` | float | Logprob of the sampled token |
| `top` | list[dict] | Top-K competing tokens and their logprobs |

### Existing Outputs

Interrupted runs are not resumed. If the output path already exists, the
script raises `FileExistsError`. Pass `--overwrite` to replace the existing
output and restart from the beginning.

## Extracted Analysis Data

Create lightweight Parquet files for repeated analysis:

```bash
python translation_analysis/data_extraction.py
```

The output mirrors the raw directory structure:

```text
translation_analysis/extracted_outputs/prompt2/CHA-Gen/{model_tag}.parquet
```

Each row represents one translation candidate. The Parquet files retain source
metadata, raw and conservatively processed text, processing status, and finish
reason. Token IDs and token-level logprobs remain only in the raw `.pkl.gz`
archive. Existing Parquet files are skipped unless `--overwrite` is passed.

### Reading Example

```python
import gzip
import pickle

with open(
    "translation_analysis/translation_outputs/prompt2/CHA-Gen/output.pkl.gz",
    "rb",
) as f:
    while True:
        try:
            with gzip.GzipFile(fileobj=f, mode="rb") as gz:
                record = pickle.load(gz)
                if record["type"] == "metadata":
                    config = record["config"]
                elif record["type"] == "batch":
                    for sample in record["samples"]:
                        src = sample["src"]
                        translations = [c["text"] for c in sample["candidates"]]
        except EOFError:
            break
```
