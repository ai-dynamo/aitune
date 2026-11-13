---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: LTX 2.3 outpaint mini benchmark
---

# LTX 2.3 outpaint with AITune and FlexTensor

This is a small, real pretrained outpaint workload for exercising the integration on a single GPU. It uses all 48
transformer blocks, the pinned published outpaint LoRA in both diffusion stages, and ordinary AITune single-Inductor
tuning with correctness checks and real eager/candidate profiling. No pretrained transformer weights are quantized,
replaced with random weights, or omitted to make the workload fit.

The 17-frame source is about 0.71 seconds at 24 FPS. Its 160x96 image is centered on an exact-black 256x160 first-stage
canvas; spatial refinement produces 512x320 output. The zero border is added in memory after video decoding because
the published adapter uses exact zeros as its mask sentinel. Source footage is the
[public-domain Helen Hill interview](https://commons.wikimedia.org/wiki/File:HelenHillInterview.theora.ogv).

`--width` and `--height` set the stage-1 canvas (positive multiples of 32); final output is twice each dimension.
The source keeps the same placement ratios: width `5/8` and height `3/5` of that canvas, rounded down to whole pixels
and required to be even for H.264. For example, `--width 768 --height 480` prepares a 480x288 source and generates
1536x960 output. Use those arguments consistently for source preparation and every run, with a separate `--work-dir`
and profile so existing mini measurements are preserved. Model weights, all 48 blocks, schedules and guidance are unchanged.

The lower resolution, duration and explicit guidance settings are this benchmark's workload choices, not claims of
equivalence to the older 45-second experiment. Both stages retain their pinned distilled sigma schedules. Modality
guidance is explicitly 1, CFG is 1, and STG/rescale are 0 in every arm; the LoRA remains active during refinement, without
a second stage-2 reference condition. Settings and source/prompt hashes are recorded for comparison.

## Runtime and memory

The tested development base is the repository's pinned NGC Torch 2.10 / 26.01 image:

```text
ghcr.io/ai-dynamo/aitune/nvcr-torch-26.01-py3@sha256:bd4b0ca692cec0d04dcfab89cb8cd1fc22a593e070c851308361d46a05fa67fb
```

Use its CUDA forward-compatibility shell initialization where required by the host driver. The example's
`pyproject.toml` lists dependencies; Diffusers comes from commit `9159a587b1f148a2cc911deb72b6d755508ef38b`.
Actual package/device versions are recorded in each result. PyAV supplies decoding and H.264 encoding; no system
FFmpeg executable is required.

Keep the same CUDA-compatible PyTorch build in all benchmark arms.

Small activations do not make the weights small. The text encoder download alone is about 48.75 GB; inference
components occupy about 45.13 GiB. On a 31 GiB host, encode the prompt in its own worker: the stored FP32 Gemma weights
load directly onto CUDA as BF16, producing a reusable CPU prompt artifact. Exit that worker before inference. If disk
space is limited, remove only that task cache's newly downloaded text-weight blobs after validating `prompt.pt`; they
are publicly downloadable again. Do not delete a preexisting shared model cache blindly.

Inference uses BF16 safetensors mmap loading on CPU so file-backed pages can be reclaimed. FlexTensor selects its
offloaded tensors and retains others on GPU. The default 0.65 GPU fraction is a starting policy, not a guaranteed
whole-process cap. Measure host RSS, actual managed weight residency, allocator peaks and whole-device occupancy.

The default requested `--pinned-memory-mode host_register` uses FlexTensor's supported in-place host-buffer pinning
instead of the extra buffer copy made by `torch` pinning. This reduces host-copy/allocator pressure on a 31 GiB host;
it does not omit weights or units. Use `--pinned-memory-mode torch` to request the alternative. Before loading weights,
the example checks FlexTensor's public host-registration capability and fails if the requested mode is unavailable.
This capability check is not proof that every subsequent pin succeeded. Result arguments record the requested mode;
check native provider logs for fallback warnings before claiming the effective mode. Both FlexTensor arms must request
the same pinning policy.

## Run each phase in a fresh process

From the AITune repository root, in the prepared runtime:

```bash
# Bound compiler worker memory on this 31 GiB host; use the same environment in every arm.
export TORCHINDUCTOR_COMPILE_THREADS=1
uv run --project examples/LTX23OutpaintAITuneFlexTensor python examples/LTX23OutpaintAITuneFlexTensor/outpaint.py prepare-source
uv run --project examples/LTX23OutpaintAITuneFlexTensor python examples/LTX23OutpaintAITuneFlexTensor/outpaint.py encode-prompt

# Resident eager reference: the whole transformer stays on GPU.
uv run --project examples/LTX23OutpaintAITuneFlexTensor python examples/LTX23OutpaintAITuneFlexTensor/outpaint.py run --mode eager --warmups 3 --runs 10

# Eager compute under real FlexTensor offloading; save its exact workload profile.
uv run --project examples/LTX23OutpaintAITuneFlexTensor python examples/LTX23OutpaintAITuneFlexTensor/outpaint.py run --mode baseline --warmups 3 --runs 10

# Same profile and offloading policy, all 48 original blocks tuned with Inductor.
uv run --project examples/LTX23OutpaintAITuneFlexTensor python examples/LTX23OutpaintAITuneFlexTensor/outpaint.py run --mode combined --reuse-profile --warmups 3 --runs 10

uv run --project examples/LTX23OutpaintAITuneFlexTensor python examples/LTX23OutpaintAITuneFlexTensor/outpaint.py compare
```

Defaults put caches, JSON, logs and media under `tuning/ltx23_outpaint/`, which is ignored by Git. Override `--work-dir`
and `--model-cache` together when using another location. Do not run the modes concurrently on the same GPU or reuse
one process's JIT/provider state for another mode. The manual staged benchmark is excluded from the shared export/
deployment-project runner. The JIT021 functional variants cover native eager/deferred tuning, the enabled performance
gate and retained-profile finalization. JIT022 checks native eager fallback after a genuine fullgraph compiler failure.

For a matched profile-loading control, preserve the profile-building baseline's results and rerun baseline with
`--reuse-profile` in another fresh process, using the same warmups/runs. This gives both FT arms the same startup path.
Inspect raw latency drift and setup order; a single sequential batch is not a statistically conclusive speedup claim.

The initial backend is exact `TorchInductorJitBackend(fullgraph=True, mode="default")`, not direct application-side
`torch.compile`. Ordinary correctness checks remain enabled. Diagnostic performance validation runs actual baseline
and candidate measurements without rejecting a slower candidate; end-to-end measurements decide whether the result
helps. Diagnostic mode requires all targets to select Inductor. `--performance-validation enabled` retains the ordinary
performance gate and accepts a measured eager winner. Both modes require successful Inductor builds and actual
eager/candidate profiling for every route; results retain the selected backend for each block.

The example records compiler limits before setup and after measurement without changing them. FlexTensor may adjust
its own process-wide limits during native setup; inspect the recorded values when comparing runtimes.

## Offload boundaries and finalization

FlexTensor selects the 48 original transformer blocks directly. AITune registers those same public offload-unit
instances after discovery and profiling finish. FlexTensor's native compiled-unit and replay support manages transfers
for AITune's repeated warmup, correctness and profiling calls. Original module attributes and parameter paths remain intact.

With `pipe` constructed and `cfg` holding the example's FT configuration, fresh retained-profile setup is:

```python
import flextensor as ft
from aitune.torch.jit.flextensor import register_flextensor_units

pipe.transformer = ft.offload(
    pipe.transformer,
    config=cfg,
    name="ltx23-outpaint-mini",
    compile_fn=lambda module: module,
)
manager = ft.get_offload_manager("ltx23-outpaint-mini")
# Run the normal pipeline until manager.phase.value == "inference".
registration = register_flextensor_units([manager])
```

Assign the exact returned proxy to `pipe.transformer`. Explicit profile loading uses public `ft.offload_from_profile`
instead of `ft.offload`, with the same native configuration and identity callback. Selectors address the original
transformer tree; no selector translation or additional module layer is required. Direct-root parameter/buffer
placement remains an application step before FT setup. See the [native integration guide](../../docs/guides/flextensor.md).

The example finishes FT startup, registers the original blocks in one batch, records both stages, enables deferred
tuning, and warms all serving routes before retaining or replanning the strategy.

The default `--finalization retain-profile` uses the provider's identity `compile_fn` plus view-profile mode in both
FlexTensor arms. It preserves the pre-AITune profile/strategy and does not claim compiled-timing replanning.
`--finalization replan` is a distinct experiment: after tuning, its callback executes exactly one complete transformer
forward per provider-requested iteration, using a captured representative stage-2 input. It never passes a whole
multi-step generation to the replan helper. Compare only matching finalization and memory policies.

## Measurements and manual review

The commands above use three warmups and ten measurements; CLI defaults of one warmup and three measurements are for
quick smoke runs. Each arm separates setup from measured generations. CUDA synchronization brackets
timing. Results record raw latencies, allocator peaks, sampled whole-device memory, process RSS, actual managed provider
memory and per-block backend/profiling telemetry. Shared disk caches make setup order-dependent; do not present it as
cold-start compilation time. A memory sample is not an exact transient high-water mark.

A separate seed not seen during tuning generates held-out final video latents and audio. `compare` reports finite
checks, absolute/relative differences, RMSE, cosine and explicit allclose diagnostics. These are not automatic
perceptual-quality judgments and do not alter AITune's built-in checks. Inspect motion, source preservation, black-mask
boundaries and consistency yourself. A visual pass is not a claim of numerical equivalence.

Native reference conditioning guides generation; it does not hard-copy the source pixels into the output. Review
source details in the resident-eager result too, and do not attribute a change to tuning merely because it is visible
in the combined video. The no-reference second stage with the outpaint LoRA still active is an explicit historical
workload choice, not a claim that every outpaint recipe uses identical conditioning.

The main review artifacts are:

- `source.mp4`: actual short source input.
- `combined/outpaint.mp4`: tuned/offloaded output for that input and tuning seed.
- `eager/outpaint.mp4` and `baseline/outpaint.mp4`: reference outputs, when completed.
- `combined/heldout_outpaint.mp4`: independent-seed visual check.
- `comparison.json` and each mode's `results.json`: actual measurements and diagnostics.

Do not infer completion or gains from these expected filenames. Inspect the results from the current run. Historical
profiles and checkpoints from the former wrapper layout are not automatically remapped to the native model tree.
There is no standalone checkpoint export, reset or rollback machinery for this integration.

Historical validation machine settings, native/CPU commands and benchmark arguments are recorded in
[`validation_commands.md`](../../work_logs/flextensor/validation_commands.md). Reconstructed commands and missing
fresh-base bootstrap history are explicitly distinguished from commands/outcomes retained from actual execution.
Those records predate this migration and do not establish native verification results.
