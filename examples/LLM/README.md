---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Tune LLMs on one or multiple GPUs"
---

This example tunes a Hugging Face causal language model with NVIDIA AITune. It runs the same generation before and
after tuning so you can verify the output, and it supports both a single GPU and Transformers native tensor
parallelism.

## What this example demonstrates

- Tune `Qwen/Qwen3-0.6B` on one GPU.
- Tune the same model across multiple GPUs with `tp_plan="auto"`.
- Select cache-aware backends for prefill and decode graphs.
- Compare generation from the original and tuned models.
- Benchmark eager and AITune execution on one GPU.

## Environment setup

Run the following commands from `examples/LLM`.

With a virtual environment managed by you:

```bash
pip install --extra-index-url https://pypi.nvidia.com .
```

With `uv`:

```bash
uv sync
```

## Tune on one GPU

```bash
uv run tune --model_id Qwen/Qwen3-0.6B --max-new-tokens 32
```

The command:

1. Loads the original model and generates a reference response.
2. Loads a fresh model, wraps it with AITune, and records prefill and decode graphs.
3. Tunes the recorded graphs with cache-appropriate backends.
4. Generates the same response with the tuned model and prints both outputs.

The default static-cache configuration uses TorchEager for prefill and TorchInductor for decode. Use `--cache` to
exercise the other configurations:

- `static`: TorchEager prefill and TorchInductor decode.
- `dynamic`: TorchEager for the recorded graph.
- `no_cache`: TorchInductor without a KV cache.

## Tune on multiple GPUs

Launch one process per GPU and add `--multi-gpu`:

```bash
uv run torchrun --standalone --nproc-per-node=4 \
  --log-dir logs --tee 3 --local-ranks-filter 0 \
  --module llm.tune \
  --multi-gpu --model_id Qwen/Qwen3-0.6B --cache dynamic --max-new-tokens 32
```

The application initializes the NCCL process group and Transformers shards the model with its native tensor-parallel
plan. AITune detects the existing distributed environment and coordinates tuning across all ranks. Rank 0 owns the
user-visible output; every rank participates in model loading, tuning, and generation. The multi-GPU example uses the
dynamic cache because Transformers static-cache generation has
[known multi-GPU compatibility issues](https://github.com/huggingface/transformers/issues/32624).

The model must provide a Transformers tensor-parallel plan. `tp_plan="auto"` and `device_map` are mutually exclusive.

### Keep distributed logs readable

By default, `torchrun` sends output from every worker to the same console. The command above uses `--tee 3` to retain
stdout and stderr from every worker under `--log-dir`, while `--local-ranks-filter 0` keeps the interactive console
focused on local rank 0. On a multi-node run, local rank 0 from each node remains visible. This launcher-level approach
keeps application logging unchanged and preserves per-rank diagnostics for investigation.

## Tune options

- `--model_id`: Hugging Face model name or path (default: `Qwen/Qwen3-0.6B`).
- `--cache`: `static`, `dynamic`, or `no_cache` (default: `static`).
- `--max-new-tokens`: Number of tokens generated during comparison and tuning (default: `512`).
- `--multi-gpu`: Enable Transformers native tensor parallelism.

## Benchmark on one GPU

Benchmark one configuration:

```bash
uv run benchmark --model_id Qwen/Qwen3-0.6B \
  --sequence_lengths "128,1024" --scenario aot --cache static
```

Compare the supported static-cache scenarios and write the results to `benchmark_all.csv`:

```bash
uv run benchmark-all --model_id Qwen/Qwen3-0.6B
```

Add `--run_baseline` to include eager generation without a KV cache. To diagnose Torch recompilation, enable its
recompilation log:

```bash
TORCH_LOGS="recompiles" uv run benchmark-all
```

The benchmark helpers currently run on one GPU; the multi-GPU command above demonstrates distributed AITune tuning
and inference.

## How cache-aware tuning works

Generation produces different compute graphs for prompt prefill and token-by-token decode. AITune records their tensor
shapes and non-tensor arguments separately, including `cache_position` and KV-cache inputs. It can therefore assign a
different strategy to each graph instead of treating the model as one static callable.

Static-cache prefill has a varying prompt length, so this example keeps it in TorchEager. Decode uses a single-token
input with a compile-friendly static cache, so the example tunes it with TorchInductor.
