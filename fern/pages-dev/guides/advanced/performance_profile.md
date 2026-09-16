---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Performance Profile"
---

After tuning a model, AITune's performance profile answers four runtime-attribution questions:

- How much runtime is inside AITune-tuned modules?
- How much runtime is inside other model modules that AITune did *not* tune?
- How much runtime is *outside* any module (pipeline glue, scheduler math, framework dispatch)?
- What do the underlying PyTorch operators and CUDA kernels look like?

The report combines PyTorch Profiler data with region annotations emitted from forward hooks on instrumented modules and from monkey-patched wrappers on their user-defined entry-point methods. Two kinds of regions are tracked: AITune-managed (AOT-wrapped) modules, and untuned modules discovered by walking the profiled object's submodule tree (skipping AITune-managed and JIT-managed subtrees). The API returns an in-memory profile with structured data and a Markdown renderer. Users choose whether and where to persist JSON or Markdown; the raw PyTorch Profiler Chrome trace is exported only when you pass an explicit `trace_file`.

The report is **factual**: it summarizes measured runtime, not recommendations.

## Quick Start

The typical flow is to profile *before* tuning (to see where time goes today and pick high-value tuning targets) and *after* tuning (to verify what actually moved). Both reports use the same call.

```python
import json
from pathlib import Path

import torch
from diffusers import DiffusionPipeline

import aitune.torch as ait
from aitune.torch.backend import TorchEagerBackend

# 1. Load the pipeline you want to profile.
pipe = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

inputs = [{"prompt": "A beautiful landscape with mountains and a lake"}]


def infer(prompt):
    return pipe(prompt, width=512, height=512, num_inference_steps=5)


# 2. (Optional) Baseline report before tuning. Discovered `nn.Module` entries
#    surface as `untuned_module:*` regions; anything else (plain `list` /
#    `dict` containers, root methods, pipeline glue) lands in residual. Useful
#    for seeing where time goes today and picking what to tune.
baseline = ait.profile(
    obj=pipe,
    input_data=inputs[0],
    inference_function=infer,
)
Path("baseline_profile.json").write_text(json.dumps(baseline.data, indent=2) + "\n")
Path("baseline_profile.md").write_text(baseline.markdown())

# 3. Inspect, wrap, and tune the modules that dominate runtime.
#    `TorchEagerBackend` is the no-compile baseline used here for a runnable
#    example; swap in `TensorRTBackend` / `TorchInductorBackend` / etc. for
#    real acceleration.
modules_info = ait.inspect(pipe, inputs, inference_function=infer)
modules = modules_info.get_modules(min_execution_percentage=0.05)
strategy = ait.OneBackendStrategy(TorchEagerBackend()).enable_find_max_batch_size(False)
pipe = ait.wrap(pipe, modules, strategy=strategy)
ait.tune(infer, inputs, batch_sizes=[1])

# 4. Profile the tuned pipeline. Tuned modules now appear as `aot_module:*`
#    regions; the rest stay as `untuned_module:*`. Comparing against the
#    baseline shows what tuning moved.
tuned = ait.profile(
    obj=pipe,
    input_data=inputs[0],
    inference_function=infer,
    warmup_runs=3,
    measured_runs=10,
    trace_file="tuned_trace.json",
)

Path("tuned_profile.json").write_text(json.dumps(tuned.data, indent=2) + "\n")
Path("tuned_profile.md").write_text(tuned.markdown())
print(tuned.trace_file.name)  # tuned_trace.json (full absolute path on .trace_file)
```

Each returned profile's `.data` field (e.g. `baseline.data`, `tuned.data`) is the JSON-ready `dict` if you want to query in-process or persist it yourself.

Step 2 is optional but recommended for an unfamiliar pipeline: without any AOT-managed modules, the report attributes discovered module entries via untuned-module discovery (including method-style entry points like `vae.decode`) and leaves non-module / framework / pipeline-glue code in residual, so you can see the shape of inference before deciding what to tune. After tuning, the same call produces a report where the tuned modules are visible as `aot_module:*` regions.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `obj` | `Any` | The model or pipeline whose live AITune metadata should be inspected. Used as the callable when `inference_function` is not provided. |
| `input_data` | `Any` | One representative input. A mapping is passed as kwargs, a tuple as positional args, `None` as no arguments, and any other value as one positional argument. |
| `inference_function` | `Callable \| None` | Optional callable to run instead of `obj(input_data)`. Useful when the model is wrapped in pipeline logic (e.g., Diffusers). |
| `warmup_runs` | `int` (default `3`) | Unmeasured warmup iterations during which the profiler is idle. One additional implicit warmup run inside the profiler absorbs its buffer-allocation cost. |
| `measured_runs` | `int` (default `10`) | Recorded iterations. |
| `trace_file` | `str \| Path \| None` | Optional Chrome trace output path. No trace is written when omitted. |

Warmup and measured invocations run under `torch.no_grad()`, matching AITune inspection and tuning.

## What's in the report

The JSON has stable top-level keys:

```json
{
  "schema_version": 1,
  "created_at": "...",
  "aitune_version": "...",
  "config": { "warmup_runs": ..., "measured_runs": ..., "uses_inference_function": ... },
  "target": { "type": "..." },
  "input": { "args_count": ..., "kwargs": [...] },
  "runs": [...],
  "profiler": { "activities": [...], "key_averages": {...} },
  "regions": [...],
  "warnings": [...]
}
```

### Per-run timing and attribution

Each entry in `runs` covers one measured iteration:

```json
{
  "run_index": 0,
  "timing": { "wall_time_us": ..., "cpu_time_us": ..., "device_time_us": ... },
  "regions": [
    { "region_id": "aot_module:unet", "calls": 6,
      "cpu_time_us": ..., "cpu_time_fraction": ...,
      "device_time_us": ..., "device_time_fraction": ... }
  ],
  "residual": {
    "cpu_time_us": ..., "cpu_time_fraction": ...,
    "device_time_us": ..., "device_time_fraction": ...
  }
}
```

- `timing` — wall-clock measurement (Python-side `time.perf_counter_ns`) plus PyTorch Profiler's CPU and device timings for the run.
- `regions` — per-region aggregates for every region observed during the run (both AOT-managed and untuned). `calls` aggregates multiple invocations within one measured run (e.g., a UNet called 6 times across denoising steps appears as one row with `calls: 6`).
- `residual` — time spent **outside** any region — neither AOT-managed nor untuned — per time domain. Negative residuals from float precision or async timing are clamped to zero.

Fractions are run-relative. Per-domain math: `sum(regions[].cpu_time_fraction) + residual.cpu_time_fraction ≈ 1.0`.

### Region metadata

`regions` at the top level describes each region observed across the run. Two kinds are emitted:

```json
[
  { "id": "aot_module:unet", "name": "unet", "kind": "aot_managed_module",
    "module_type": "diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
    "wrapper_state": "tuned" },
  { "id": "untuned_module:text_encoder", "name": "text_encoder", "kind": "untuned_module",
    "module_type": "transformers.models.clip.modeling_clip.CLIPTextModel" }
]
```

- `aot_managed_module` — an AITune-wrapped module. Carries `wrapper_state` reflecting the live wrapper FSM (`tuned`, `recording`, `passthrough`). If a region was observed in the profile but the wrapper is no longer in `MODULE_REGISTRY` (e.g., it was unwrapped between profiling and report assembly), the state is `"unknown"` and an `UNREGISTERED_AOT_REGION` warning is emitted.
- `untuned_module` — an `nn.Module` reachable from `obj` that is not AITune-managed and not JIT-patched. Discovered recursively, with subtrees containing managed descendants descended-into rather than attributed as a whole (so siblings of a tuned module still get their own region). `wrapper_state` is omitted — the field is AOT-specific.

#### Method-style entry points

Some pipelines invoke an `nn.Module` via a custom method rather than via `__call__` — e.g., Diffusers' `pipe.vae.decode(z)`, or HuggingFace `model.generate(...)` when the model is held as an attribute of the profiled object rather than being the profiled object itself (see the root-method caveat below). Forward hooks alone never see those calls. Discovery additionally wraps user-defined methods declared on the module's class hierarchy (anything beyond what `nn.Module` itself provides, excluding dunders, `forward`, and non-function descriptors). Each wrapped method produces a region whose path is `<module_path>.<method_name>`:

```json
{ "id": "untuned_module:vae.decode", "name": "vae.decode", "kind": "untuned_module",
  "module_type": "diffusers.models.autoencoders.autoencoder_kl_ltx2.AutoencoderKLLTX2Video" }
```

Method-region `module_type` is the type of the underlying module (the parent path). Method wrappers preserve the original signature (`inspect.signature` round-trips) so framework code that introspects methods sees the real shape with the wrapper active. Methods that aren't actually invoked during measurement produce no region rows.

This applies to *untuned* targets only. AOT-managed modules receive forward hooks but not method wrappers — wrapping a tuned module's methods would overlap with its own forward span when the method internally calls `__call__`.

### PyTorch Profiler drilldown

`profiler.key_averages` contains two bounded sorted views (`cpu_time_total` and `device_time_total`) of the profiler's own per-operator aggregates. Each row aggregates across all `measured_runs` iterations combined:

```json
{
  "key": "aten::scaled_dot_product_attention",
  "device_type": "CPU",
  "count": 723,
  "self_cpu_time_total_us": ..., "cpu_time_total_us": ...,
  "self_device_time_total_us": ..., "device_time_total_us": ...
}
```

Use this to find the dominant operators inside (or outside) tuned regions. The view is bounded to 20 rows per sort key to keep the JSON compact; when `trace_file` is provided, the full event stream lives in that Chrome trace.

## The Markdown view

`profile.markdown()` renders a view over `profile.data` for readability. It contains:

- **Overview** — created timestamp, AITune version, target, configured warmup/measured counts, input positional-arg count and keyword-arg names.
- **Runs** — per-run wall, CPU, and device timing.
- **Regions** — region metadata table (name, kind, wrapper state, module type). For untuned regions, the State column is `-`.
- **Per-Run Attribution** — region + residual table. The `_(residual)_` row appears after each run's regions so you can visually verify the column sums to ~100%.
- **Profiler Key Averages** — top operators by CPU total and device total.
- **Warnings** — structured warnings (see below).

The Markdown is a pure view; rerunning `profile.markdown()` against the same data always produces identical output.

## Interpreting residual

Residual is the time NOT covered by any region — neither AOT-managed nor untuned. Discovery walks the submodule tree of `obj` reached via direct `nn.Module` attributes and transparent `nn.ModuleList` / `nn.ModuleDict` containers; **plain Python `list` and `dict` attributes that happen to hold modules are not entered**. In typical pipelines this covers everything, and the residual is almost entirely code that runs *outside* any `nn.Module.forward`:

- Pipeline-level Python (e.g., a diffusion `__call__` running `scheduler.step()`, classifier-free guidance combine, latent scaling, image post-processing).
- Standalone tensor ops between module calls (`torch.cat`, `.chunk`, arithmetic on logits/latents).
- Host-side CUDA driver work — `cudaLaunchKernel`, `cudaMemcpyAsync`, `cudaStreamSynchronize` (the last typically triggered by `.item()` calls inside scheduler math).
- Framework dispatch (PyTorch's overhead between operator calls).

If a pipeline stores modules in a plain `list` or `dict` rather than a `ModuleList` / `ModuleDict`, those modules will not be discovered and their compute will land in residual. Convert to a `ModuleList` / `ModuleDict` (the PyTorch-idiomatic choice anyway) to get them attributed.

On a Stable Diffusion v1.5 pipeline with UNet + VAE decoder AOT-wrapped (and `text_encoder`, `safety_checker`, `vae.post_quant_conv` automatically captured as untuned regions), you typically see:

- **CPU residual ~28%** — scheduler math, CFG combine, latent prep, image post-processing, host-side CUDA driver overhead.
- **Device residual ~0.2%** — essentially all GPU work lives inside a region.

A large CPU residual on a pipeline-heavy workload is expected (Python orchestration is real). A large *device* residual means there's significant GPU work happening outside any captured `nn.Module.forward` — that's unusual and worth investigating in the Chrome trace.

Residual is computed only for CPU and device. Per-region wall time isn't available from PyTorch Profiler, so `residual` doesn't include a wall figure.

## Configuration tips

### Warmup

The default `warmup_runs=3` works for most real-sized models. AITune always runs one *additional* warmup iteration inside the profiler to absorb buffer allocation, so a value of `0` is permitted and still produces a clean first measured run — but pure model warmup (CUDA kernel autotune, cuDNN heuristics, allocator state) benefits from a few iterations before measurement.

For sub-millisecond workloads where micro-warmup effects on the CPU side are visible (branch predictor, cache state), increase to `warmup_runs=5` or higher. On real inference models (tens to hundreds of milliseconds), `warmup_runs=3` is plenty.

### Measured runs

`measured_runs=10` gives stable averages on most workloads. Increase if you see run-to-run variance you want to characterize statistically. The cost is roughly linear: 10 measured runs of a 100 ms inference takes ~1 second to profile plus negligible analysis overhead.

## Caveats

- **Single-input profiling.** V1 profiles one representative input scenario, repeated. Multi-scenario aggregation (different batch sizes, different prompts) is future work.
- **CUDA-side device time discrepancy.** In `profiler.key_averages`, you may see two entries per AITune region — one with `device_type: "CPU"` and one with `device_type: "CUDA"`. The CPU-side row's `device_time_total` is the sum of kernel device times launched *within* the region. The CUDA-side row's `device_time_total` is the GPU-timeline span of the region, including idle gaps between kernels. Our per-region `device_time_us` in `runs[]` uses the CPU-side view (sum of kernels), which is the right value for attribution.
- **Trace size.** Chrome traces can be tens of megabytes for typical models. Compress before sharing if needed.
- **AOT-wrapped modules that are not yet tuned.** AITune wrappers in `recording` or `passthrough` state still emit their AOT region annotation. You can therefore run `aitune.torch.profile(...)` *before* tuning to scope candidate modules; the region appears with `wrapper_state` set to `"recording"` or `"passthrough"` rather than `"tuned"`. (Distinct from the `untuned_module` kind, which refers to modules AITune never wrapped at all.)
- **Methods on a bare `nn.Module` object are not instrumented.** When `obj` is itself an `nn.Module`, discovery starts at `obj.named_children()` — so methods on the root (e.g. `model.generate`) are *not* wrapped, and their pipeline-glue cost (loops, sampling, KV cache management) lands in residual. Per-child compute is captured correctly. To attribute root-level methods, wrap the model in a small pipeline-like container holding it as an attribute and pass that container as `obj` (`Container(model=my_model)` → `obj=container`). The cleaner fix needs hierarchical attribution and is tracked as follow-up work.
- **Pre-bound method references bypass the wrapper.** Method wrapping monkey-patches at the *instance attribute* level. A reference captured before `aitune.torch.profile(...)` enters (`saved = pipe.vae.decode`) won't go through the wrapper if invoked later. Pipeline code that resolves the attribute at call site (`self.vae.decode(...)`) is captured normally — this is the common case.
- **Direct calls into descendants of a clean untuned parent are missed.** Untuned-module discovery stops at the topmost clean module (no AITune-managed descendants) and installs hooks only there. If pipeline code skips the parent and calls a deeper submodule directly (`self.block.inner(x)` instead of `self.block(x)`), the parent hook never fires and the deeper submodule was never instrumented — the call's compute lands in residual. Per-event nested-untuned suppression (the inverse of the existing AOT case) would let us instrument at every level safely; tracked as follow-up.

## Warnings

Structured warnings appear in the `warnings` array with stable codes:

| Code | Source | Meaning |
|---|---|---|
| `UNMAPPED_AOT_REGION_EVENTS` | `core` | An AITune region event was observed in the profiler stream with no profiled-run ancestor. Indicates a data-integrity issue, usually benign at low counts. |
| `UNREGISTERED_AOT_REGION` | `core` | A region observed in the profile is not present in the live `MODULE_REGISTRY`. The region appears in the report with `wrapper_state: "unknown"`. |

## Relationship to other observability tools

Runtime attribution is **complementary** to AITune's other observability features:

| Feature | Question it answers |
|---|---|
| `aitune.torch.profile(...)` | *Where is time spent in my code?* |
| [`ModuleFunctionKernelProfiler`](module_function_kernel_profiler.md) | *Which module-scoped `torch.nn.functional` calls launch each CUDA kernel, and with which inputs?* |
| `AITUNE_HARDWARE_METRICS=1` | *How is the hardware behaving?* (GPU/CPU utilization, memory, power) |
| `AITUNE_NVTX_EVENTS=1` | *What does this look like in Nsight Systems?* |
| Tuning telemetry | *What did AITune choose and why?* |

They measure different things on different time models. For a complete diagnostic view, enable all three with their respective env vars.

The Chrome trace at `profile.trace_file`, when requested, opens directly in [Perfetto](https://ui.perfetto.dev/) or `chrome://tracing` for kernel-level inspection — useful when the JSON summary points to a region of interest and you want to drill into individual operators.
