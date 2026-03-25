---
name: aitune-tune
description: Use when a user asks to tune, optimize, accelerate, or deploy a PyTorch model or inference pipeline for GPU performance.
license: Apache-2.0
---
<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# AITune Model Tuning (Agent Skill)

Tuning is the process of finding the best backend for a model/pipeline. It iterates backends from highest to lowest performance, stopping at the first that compiles correctly and meets the speedup threshold.


# Tuning Modes

Choose the mode based on user requirements — these are parallel paths, not sequential steps.

| Mode | When to Use | Guide |
|---|---|---|
| **JIT** | Zero code changes, quick experiments, unknown batch sizes | `how-to-jit-tune.md` |
| **AOT** | Production, benchmarking, checkpoint persistence, explicit shape control | `how-to-aot-tune.md` |

The workflow below (Phases 1–5) is the **AOT path**. For JIT, load `how-to-jit-tune.md` instead.

# Backends

| Backend | Best For | Notes |
|---|---|---|
| `TensorRTBackend` | Production, highest performance | fp16/fp8/int8, CUDA Graphs |
| `TorchTensorRTAotBackend` | AOT via `torch_tensorrt.compile` | Good graph break tolerance |
| `TorchTensorRTJitBackend` | JIT via `torch.compile` | Most PyTorch-compatible |
| `TorchAOBackend` | PyTorch-native optimization | No config needed |
| `TorchInductorBackend` | PyTorch compiler optimization | No config needed |
| `Vanilla PyTorch` | Baseline / fallback | No optimization |

## TensorRT Backend Configuration

```python
from aitune.torch.backend import TensorRTBackend, TensorRTBackendConfig

config = TensorRTBackendConfig(quantization_config=ONNXAutoCastConfig(), use_cuda_graphs=True, use_dynamo=True)
backend = TensorRTBackend(config)
```

# Tuning Strategies

| Strategy | Behavior |
|---|---|
| `OneBackendStrategy` | Use exactly one specified backend |
| `FirstWinsStrategy` | Use the first backend that compiles successfully |
| `HighestThroughputStrategy` | Profile all backends, select the fastest |


# Common Issues

| Issue | AOT Behavior | JIT Behavior |
|---|---|---|
| Graph breaks | Must be handled manually | Skips problematic modules automatically |
| Dynamic shapes | Detects and configures axes explicitly | Uses shapes seen at runtime only |
| Batch size variation | Can extrapolate via dynamic axes | Limited to observed batch sizes |
| TRT static engine for variable-length inputs | Provide input_data with multiple samples of different sequence lengths so TRT builds min/opt/max profiles; a single-length sample produces a static engine that fails at inference time on other lengths | N/A |
| TorchTRT-JIT + int64 embedding indices | N/A (not a JIT path) | TRT requires int32 for embedding indices; models using int64 token IDs (all HuggingFace transformers) will fall back to eager — skip TorchTRT-JIT for pure embedding/encoder models |
| TorchAO fp8 on non-Hopper GPU | fp8e4nv requires SM ≥ 90 (H100+); skip TorchAO when `torch.cuda.get_device_capability()[0] < 9` | Same |
| TorchInductorBackend with read-only triton cache | Set `TRITON_CACHE_DIR` env var to a writable path before running; treat this as an environment fix, not a backend failure — retry the same backend after fixing | Same |


# Key Source Paths

- `aitune/torch/` — Main AOT tuning API (`inspect`, `wrap`, `tune`, `save`, `load`)
- `aitune/torch/jit/` — JIT tuning implementation and config
- `aitune/torch/backend/` — All backend implementations
- `aitune/torch/tune_strategy/` — Strategy implementations
- `examples/` — Working examples: LLM, ResNet, StableDiffusion, FLUX, Parakeet, ESM2, E5Large

# Workflow - Execution Patterns

## Prerequisites

- Create and use one directory in current working directory i.e. `tuning/<model_name>` for the tuning process.
- Find and install dependencies for the model/pipeline.
- Use input data provided by the user, use huggingface examples and as a last resort use the default input data.
- Try to construct the input of batch size 2 for the model/pipeline to be able to detect the batch dimensions.
- Try not to disaggregate the pipeline - do not create submodules glue code - keep the pipeline as one unit.
- To make output concise try block progress bars for pipelines with `pipe.set_progress_bar_config(disable=True)` or `TQDM_DISABLE=1` environment variable etc.
- To get tuning logs try to set `AITUNE_CONSOLE_OUTPUT=1` environment variables
- Create a new cache directory for the tuning process and set it as `AITUNE_CACHE_DIR` environment variable.
- Create a new triton cache directory and set it as `TRITON_CACHE_DIR` environment variable.
- Do the environment setup checklist before running the tuning script.

### Environment Setup Checklist

Before running any tuning script, verify:

```bash
# GPU is visible
nvidia-smi

# CUDA available in Python and GPU SM version (needed for fp8 gating)
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, 'SM', torch.cuda.get_device_capability())"

# AITune importable
python -c "import aitune.torch as ait; print('ok')"

# TensorRT importable (for TRT backends)
python -c "import tensorrt; print(tensorrt.__version__)"

If any other check fails, stop and report the blocker before attempting tuning.


## Phase 1: Inspecting a model

Use the `aitune-inspect` skill to inspect the model provided by the user and get the names and utilization of the submodules.

Capture the output and use it to populate the Tuning Summary.

## Phase 2: Wrapping with a specific backend

### Example configurations
Below, there are all backends configurations for the all optimization steps.

```python
from aitune.torch.backend import (
    TensorRTBackend, TensorRTBackendConfig,
    TorchTensorRTJitBackend, TorchTensorRTAotBackend,
    TorchAOBackend, TorchInductorBackend,
)
from aitune.torch.tune_strategy import OneBackendStrategy, HighestThroughputStrategy

# TensorRT fp16 (default production choice)
cfg = TensorRTBackendConfig(quantization_config=ONNXAutoCastConfig(), use_cuda_graphs=True, use_dynamo=True)
strategy = OneBackendStrategy(TensorRTBackend(cfg))

# TensorRT fp32 (fallback if fp16 correctness fails)
cfg = TensorRTBackendConfig()
strategy = OneBackendStrategy(TensorRTBackend(cfg))

# TorchTRT AOT
strategy = OneBackendStrategy(TorchTensorRTAotBackend())

# TorchTRT JIT (torch.compile path)
strategy = OneBackendStrategy(TorchTensorRTJitBackend())

# TorchAO
strategy = OneBackendStrategy(TorchAOBackend()

# Torch Inductors JIT and AOT
strategy = OneBackendStrategy(TorchInductorAotBackend())
strategy = OneBackendStrategy(TorchInductorJitBackend())


modules = ait.inspect(model, input_data).get_modules()
model = ait.wrap(model, modules, strategy=strategy)
ait.tune(model, input_data)
```

### Depth-Scoped Wrapping

The caller specifies the module scope. Use the appropriate `inspect()` call:

```python
# Root scope — entire pipeline as one unit
modules_info = ait.inspect(model, input_data, min_depth=0)

# Depth=1 scope — immediate submodules only
modules_info = ait.inspect(model, input_data, min_depth=1)

modules = modules_info.get_modules()
model = ait.wrap(model, modules, strategy=strategy)
ait.tune(model, input_data)
```

If the caller (e.g. tuning-assistant agent) specifies a backend trial order, follow that. The Default Backend Trial Order below applies when running standalone.

### Backend Selection

Use the inspection output to determine trial order. Do not invent an order — use the table below.

| Condition | Trial Order |
|---|---|
| Production, fp16 viable | TRT(fp16) → TRT(fp32) → TorchTRT-AOT → TorchTRT-JIT → TorchAO → Inductor |
| PyTorch compatibility required | TorchTRT-JIT → TorchAO → Inductor → TRT(fp16) |
| Quantization needed (fp8/int8) | TRT(fp8) → TRT(int8) → TRT(fp16) |
| Quick experimentation / JIT mode | JIT via env var — no AOT loop needed |

Declare the trial order explicitly before starting Phase 3.


### Default Backend Trial Order

When no specific requirements are given, try in this order and stop at the first GO result:

1. `TensorRTBackend(fp16, dynamo=On/Off)` — highest performance
2. `TensorRTBackend(fp32, dynamo=On/Off)` — if fp16 correctness fails
3. `TorchTensorRTAotBackend` — good perf, better graph break tolerance
4. `TorchTensorRTJitBackend` — torch.compile path, most PyTorch-compatible
5. `TorchAOBackend` — PyTorch-native, no TRT required
6. `TorchInductorAotBackend` — Torch Inductor AOT backend
6. `TorchInductorJitBackend` — Torch Inductor JIT backend
7. Vanilla PyTorch - Fallback baseline path


If TRT correctness fails with `use_dynamo=True`, retry with `use_dynamo=False` before moving to the next backend.

## Phase 3 — Optimization Loop (iterate until working backend is found)

This is the core loop. Run it for each backend in the declared trial order. Stop as soon as a backend produces a **GO** result. If all backends are exhausted, produce a **NO-GO** report and move to Phase 4.

### Step 3a — Tune with the selected backend

Check examples and `how-to-aot-tune.md` to create the tuning script. Capture the output and use it to populate the Tuning Summary.

Distinguish failure types:
- **Compile failure** (backend rejects the model): advance to next backend in trial order
- **Environment failure** (read-only triton cache, missing permissions, writable path issues): fix the environment condition and **retry the same backend** before advancing — do not count environment failures as backend failures
- **Unhandled exception / script crash** (OOM, CUDA error, import error): diagnose using `common-errors.md` before advancing

### Step 3b — Check correctness validation

Use the `aitune-validate` skill to validate the output of the tuned model against the baseline model.

### Step 3c — Benchmark

Use the `aitune-benchmark` skill to benchmark the tuned model.



#### Loop stopping criteria

| Condition | Action |
|---|---|
| `compile_ok AND speedup >= 1.1 AND correctness_pass` | **STOP** — this backend wins. Save checkpoint, proceed to Phase 4. |
| `compile_ok AND 1.0 <= speedup < 1.1 AND correctness_pass` | Log as CONDITIONAL candidate. Continue trying remaining backends for better speedup. |
| Compile failed OR correctness failed OR speedup < 1.0 | Advance to next backend. |
| All backends exhausted, no winner | Proceed to Phase 4 with best CONDITIONAL candidate, or NO-GO if none. |

After each backend attempt, print a one-line status:
```
[Backend: TensorRTBackend-fp16] compile=OK  speedup=2.1x  correctness=PASS  → WINNER
[Backend: TorchTRT-AOT]        compile=FAIL (graph break in layer X)        → SKIP
```


## Phase 4 — Save Checkpoint

If a winning backend was found, save the tuned model and verify the file exists:

```python
import json, os, aitune.torch as ait

checkpoint_path = "tuned_model.ait"
ait.save(model, checkpoint_path)

print(json.dumps({
    "checkpoint": checkpoint_path,
    "saved": os.path.exists(checkpoint_path),
    "size_mb": os.path.getsize(checkpoint_path) / 1024**2 if os.path.exists(checkpoint_path) else 0,
}))
```


## Phase 5 — Deployment Readiness Report

Populate from actual captured numbers. Do not estimate or approximate — every value must come from a script output.

```markdown
## Tuning & Deployment Readiness Report

**Model**: [from inspection]
**Winning backend**: [backend name + precision]
**Backends attempted**: [count and names]
**Analysis date**: [date]


### Performance Results

| Metric          | Baseline (Eager) | Tuned   | Delta    |
|-----------------|------------------|---------|----------|
| Throughput (r/s)| [value]          | [value] | [+X.Xx]  |
| Avg latency (ms)| [value]          | [value] | [-Xms]   |
| p50 latency (ms)| [value]          | [value] | [-Xms]   |
| p95 latency (ms)| [value]          | [value] | [-Xms]   |
| p99 latency (ms)| [value]          | [value] | [-Xms]   |
| GPU memory (MB) | [value]          | [value] | [+-XMB]  |

**Speedup**: [X.Xx]
**Compilation time**: [Xs]


### Correctness Validation

**Max absolute difference**: [value]
**Max relative error**: [value]
**Tolerance threshold**: [atol used]
**Result**: PASS / FAIL


### Backend Trial Log

| Backend | Compile | Speedup | Correctness | Outcome |
|---------|---------|---------|-------------|---------|
| [name]  | OK/FAIL | [X.Xx]  | PASS/FAIL   | WINNER/SKIP |


### Deployment Recommendation

**Status**: GO / CONDITIONAL GO / NO-GO

**Reasoning**: [grounded in the numbers above]

**Conditions** (if CONDITIONAL GO): [specific, actionable list]

**Checkpoint path**: [absolute path to .ait file]
```

# Common Errors and Diagnostics

There are some common errors and diagnostics that you can find in the attached file `common-errors.md` load if needed.
