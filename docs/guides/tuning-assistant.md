---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Tuning Assistant Agent"
---

The `tuning-assistant` subagent finds the fastest backend for a PyTorch model or pipeline using NVIDIA AITune. It runs the full AOT tuning loop — inspect, wrap, tune, validate, benchmark — and returns a deployment recommendation with a saved `.ait` checkpoint.

## How to invoke

In Claude Code, just mention the subagent name:

```
as tuning-assistant subagent <task description>

@tuning-assistant <task description>
```

## What to include in the prompt

| Element | Required | Notes |
|---|---|---|
| Model reference | Yes | HuggingFace model ID/URL or local path |
| Input description | Recommended | Shape, dtype, typical sequence length or image size |
| Batch sizes | Optional | Default: tries batch=2 for shape detection |
| Precision preference | Optional | Default: fp16 first, fp32 fallback |
| Special constraints | Optional | e.g. "no TRT dependency", "max 4GB GPU memory" |

For HuggingFace models the agent can infer inputs from the model card, but providing them avoids an extra round-trip.

## Example prompts

**Minimal (agent infers inputs from HF model card):**
```
as tuning-assistant subagent find the best performing backend for facebook/roberta-large
```

**Recommended (explicit input context):**
```
@tuning-assistant find the best performing backend for facebook/roberta-large
— text input, sequence length up to 512, batch size 1
```

**With constraints:**
```
@tuning-assistant find the best performing backend for facebook/roberta-large
— text input, batch 1-8, fp16 preferred, save checkpoint to tuning/roberta/
```

## What the agent does

1. Runs environment checks (GPU, CUDA, AITune, TensorRT)
2. Inspects the model — tries root module first, then depth=1 submodules if root fails
3. Iterates backends in priority order: TRT-fp16 → TRT-fp32 → TorchTRT-AOT → TorchTRT-JIT → TorchAO → Inductor → TorchEager
4. Validates correctness and benchmarks each successful backend
5. Saves a `.ait` checkpoint and reports GO / CONDITIONAL GO / NO-GO

Work is placed under `tuning/<model_name>/` in the current directory.

## Example output

```
[root | TensorRTBackend-fp16]  compile=OK  correctness=PASS  speedup=2.3x  → WINNER

## Optimization Report

Model: RobertaModel
Strategy: root
Winning backends: root → TensorRTBackend (fp16)
Checkpoint: tuning/roberta-large/tuned_model.ait

| Module | Backend           | Speedup | Notes        |
|--------|-------------------|---------|--------------|
| root   | TensorRTBackend   | 2.3x    | fp16, dynamo |

Recommendation: GO
```
