---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "LLM models tuning"
---

This example demonstrates how to use NVIDIA AITune to tune LLMs.

## Environment Setup

You can use either of the following options to set up the environment:

### Option 1 - virtual environment managed by you

Activate your virtual environment and install the dependencies:

```bash
pip install --extra-index-url https://pypi.nvidia.com .
```

### Option 2 - virtual environment managed by `uv`

Install dependencies:

```bash
uv sync
```

## Usage

### Tune

To tune an LLM model, run:

```bash
tune --model_id microsoft/Phi-3-mini-4k-instruct
```

or with `uv`:

```bash
uv run tune --model_id microsoft/Phi-3-mini-4k-instruct
```

Arguments:

- `--model_id`: Model ID to load from Hugging Face Hub (default: "microsoft/Phi-3-mini-4k-instruct")
- `--cache`: Cache implementation type, either "dynamic" or "static" (default: "static")

### Benchmark

To benchmark a specific configuration:

```bash
benchmark --model_id microsoft/Phi-3-mini-4k-instruct --sequence_lengths "128,1024"
```

or with `uv`:

```bash
uv run benchmark --model_id microsoft/Phi-3-mini-4k-instruct --sequence_lengths "128,1024"
```

Arguments:

- `--model_id`: Model ID from Hugging Face Hub
- `--sequence_lengths`: Sequence length pairs in format 'ISL,OSL' (e.g., '128,512' or '128,512 256,1024')
- `--iterations`: Number of benchmark iterations
- `--warmup_iters`: Number of warmup iterations
- `--cache`: Cache type ("no_cache", "dynamic", "static")
- `--scenario`: Benchmark scenario ("vanilla", "aot")

### Benchmark All

To benchmark all scenarios:

```bash
benchmark-all --model_id microsoft/Phi-3-mini-4k-instruct
```

or with `uv`:

```bash
uv run benchmark-all --model_id microsoft/Phi-3-mini-4k-instruct
```

Arguments:

- `--model_id`: Model ID from Hugging Face Hub
- `--sequence_lengths`: Sequence length pairs (default includes (1, 512), (1, 1024), (1, 2040))
- `--run_baseline`: Whether to run baseline scenario (no_cache, vanilla)


## Discussion

### Cache type

`TorchInductor` (and `torch.compile`) supports only `static` cache. `dynamic` cache is added only for comparison and can tuned with `TorchEager` backend.

### Detecting prefill/decode, kv cache entries

During prefill AITune detects inputs as:
```
 Tensors:
╒════════════════╤═════════════════╤═══════════════════════════════╤═══════════════╤═══════════════╤═════════════╕
│ Access Path    │ Semantic Path   │ Shape                         │ Min Shape     │ Max Shape     │ Dtype       │
╞════════════════╪═════════════════╪═══════════════════════════════╪═══════════════╪═══════════════╪═════════════╡
│ attention_mask │ attention_mask  │ ['batch0', 1, 'dim2', 'dim3'] │ [1, 1, 5, 24] │ [2, 1, 6, 25] │ torch.bool  │
├────────────────┼─────────────────┼───────────────────────────────┼───────────────┼───────────────┼─────────────┤
│ cache_position │ cache_position  │ ['dim0']                      │ [5]           │ [6]           │ torch.int64 │
├────────────────┼─────────────────┼───────────────────────────────┼───────────────┼───────────────┼─────────────┤
│ input_ids      │ input_ids       │ ['batch0', 'dim1']            │ [1, 5]        │ [2, 6]        │ torch.int64 │
├────────────────┼─────────────────┼───────────────────────────────┼───────────────┼───────────────┼─────────────┤
│ position_ids   │ position_ids    │ ['batch0', 'dim1']            │ [1, 5]        │ [2, 6]        │ torch.int64 │
╘════════════════╧═════════════════╧═══════════════════════════════╧═══════════════╧═══════════════╧═════════════╛
Other:
╒════════════════╤═════════════════╤═════════╕
│ Access Path    │ Semantic Path   │ Value   │
╞════════════════╪═════════════════╪═════════╡
│ inputs_embeds  │ inputs_embeds   │ None    │
├────────────────┼─────────────────┼─────────┤
│ logits_to_keep │ logits_to_keep  │ 1       │
├────────────────┼─────────────────┼─────────┤
│ return_dict    │ return_dict     │ True    │
├────────────────┼─────────────────┼─────────┤
│ use_cache      │ use_cache       │ True    │
╘════════════════╧═════════════════╧═════════╛
```

As can be seen it detects batch axis `batch0` and sequence length axes `dim1`, `dim2`, `dim3` (the ordinal number says at which rank the axis is). At first generation kv cache is empty because it
is lazily initialized. First prefill causes cache to be initialized and its tensors are detected in the outputs:

```
 output_spec:
 Tensors:
╒═════════════════════════════════════════════╤═════════════════════════════════════════════╤════════════════════════════╤═════════════════╤═════════════════╤════════════════╕
│ Access Path                                 │ Semantic Path                               │ Shape                      │ Min Shape       │ Max Shape       │ Dtype          │
╞═════════════════════════════════════════════╪═════════════════════════════════════════════╪════════════════════════════╪═════════════════╪═════════════════╪════════════════╡
│ output["logits"]                            │ logits                                      │ ['batch0', 1, 32064]       │ [1, 1, 32064]   │ [2, 1, 32064]   │ torch.bfloat16 │
├─────────────────────────────────────────────┼─────────────────────────────────────────────┼────────────────────────────┼─────────────────┼─────────────────┼────────────────┤
│ output["past_key_values"].layers[0].keys    │ ('past_key_values', 'layers', 0, 'keys')    │ ['batch0', 32, 'dim2', 96] │ [1, 32, 24, 96] │ [2, 32, 25, 96] │ torch.bfloat16 │
├─────────────────────────────────────────────┼─────────────────────────────────────────────┼────────────────────────────┼─────────────────┼─────────────────┼────────────────┤
│ output["past_key_values"].layers[0].values  │ ('past_key_values', 'layers', 0, 'values')  │ ['batch0', 32, 'dim2', 96] │ [1, 32, 24, 96] │ [2, 32, 25, 96] │ torch.bfloat16 │
├─────────────────────────────────────────────┼─────────────────────────────────────────────┼────────────────────────────┼─────────────────┼─────────────────┼────────────────┤
│ output["past_key_values"].layers[31].keys   │ ('past_key_values', 'layers', 31, 'keys')   │ ['batch0', 32, 'dim2', 96] │ [1, 32, 24, 96] │ [2, 32, 25, 96] │ torch.bfloat16 │
├─────────────────────────────────────────────┼─────────────────────────────────────────────┼────────────────────────────┼─────────────────┼─────────────────┼────────────────┤
│ output["past_key_values"].layers[31].values │ ('past_key_values', 'layers', 31, 'values') │ ['batch0', 32, 'dim2', 96] │ [1, 32, 24, 96] │ [2, 32, 25, 96] │ torch.bfloat16 │
╘═════════════════════════════════════════════╧═════════════════════════════════════════════╧════════════════════════════╧═════════════════╧═════════════════╧════════════════╛
```

On the first decode iteration those kv cache tensors can be seen as inputs:
```text
 Tensors:
╒═══════════════════════════════════╤═════════════════════════════════════════════╤════════════════════════════╤═════════════════╤═════════════════╤════════════════╕
│ Access Path                       │ Semantic Path                               │ Shape                      │ Min Shape       │ Max Shape       │ Dtype          │
╞═══════════════════════════════════╪═════════════════════════════════════════════╪════════════════════════════╪═════════════════╪═════════════════╪════════════════╡
│ attention_mask                    │ attention_mask                              │ ['batch0', 1, 1, 'dim3']   │ [1, 1, 1, 24]   │ [2, 1, 1, 25]   │ torch.bool     │
├───────────────────────────────────┼─────────────────────────────────────────────┼────────────────────────────┼─────────────────┼─────────────────┼────────────────┤
│ cache_position                    │ cache_position                              │ [1]                        │ [1]             │ [1]             │ torch.int64    │
├───────────────────────────────────┼─────────────────────────────────────────────┼────────────────────────────┼─────────────────┼─────────────────┼────────────────┤
│ input_ids                         │ input_ids                                   │ ['batch0', 1]              │ [1, 1]          │ [2, 1]          │ torch.int64    │
├───────────────────────────────────┼─────────────────────────────────────────────┼────────────────────────────┼─────────────────┼─────────────────┼────────────────┤
│ past_key_values.layers[0].keys    │ ('past_key_values', 'layers', 0, 'keys')    │ ['batch0', 32, 'dim2', 96] │ [1, 32, 24, 96] │ [2, 32, 25, 96] │ torch.bfloat16 │
├───────────────────────────────────┼─────────────────────────────────────────────┼────────────────────────────┼─────────────────┼─────────────────┼────────────────┤
│ past_key_values.layers[0].values  │ ('past_key_values', 'layers', 0, 'values')  │ ['batch0', 32, 'dim2', 96] │ [1, 32, 24, 96] │ [2, 32, 25, 96] │ torch.bfloat16 │
├───────────────────────────────────┼─────────────────────────────────────────────┼────────────────────────────┼─────────────────┼─────────────────┼────────────────┤
│ past_key_values.layers[31].keys   │ ('past_key_values', 'layers', 31, 'keys')   │ ['batch0', 32, 'dim2', 96] │ [1, 32, 24, 96] │ [2, 32, 25, 96] │ torch.bfloat16 │
├───────────────────────────────────┼─────────────────────────────────────────────┼────────────────────────────┼─────────────────┼─────────────────┼────────────────┤
│ past_key_values.layers[31].values │ ('past_key_values', 'layers', 31, 'values') │ ['batch0', 32, 'dim2', 96] │ [1, 32, 24, 96] │ [2, 32, 25, 96] │ torch.bfloat16 │
├───────────────────────────────────┼─────────────────────────────────────────────┼────────────────────────────┼─────────────────┼─────────────────┼────────────────┤
│ position_ids                      │ position_ids                                │ ['batch0', 1]              │ [1, 1]          │ [2, 1]          │ torch.int64    │
╘═══════════════════════════════════╧═════════════════════════════════════════════╧════════════════════════════╧═════════════════╧═════════════════╧════════════════╛
Other:
╒════════════════╤═════════════════╤═════════╕
│ Access Path    │ Semantic Path   │ Value   │
╞════════════════╪═════════════════╪═════════╡
│ inputs_embeds  │ inputs_embeds   │ None    │
├────────────────┼─────────────────┼─────────┤
│ logits_to_keep │ logits_to_keep  │ 1       │
├────────────────┼─────────────────┼─────────┤
│ return_dict    │ return_dict     │ True    │
├────────────────┼─────────────────┼─────────┤
│ use_cache      │ use_cache       │ True    │
╘════════════════╧═════════════════╧═════════╛

```

The detection of the prefill/decode phase cannot be based only by looking at kv cache entries. This is because HuggingFace caches already initialized kv cache for
subsequent calls. This is based on the `cache_position` tensor.

### Detecting recompilations

If you would like to see recompilation run `benchmark/benchmark_all` script with torch flag:


```bash
TORCH_LOGS="recompiles" uv run benchmark-all
```
