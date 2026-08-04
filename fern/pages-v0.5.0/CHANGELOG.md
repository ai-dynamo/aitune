---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Changelog"
---
# Changelog

## 0.5.0
- feat: allow AOT modules to declare explicit dynamic input shapes
- feat: Add `MinLatencyStrategy` tune strategy that profiles all backends and selects the one with the lowest latency
- feat: Add `LatencyBudgetStrategy` tune strategy that maximizes throughput while satisfying a latency budget
- feat: validate model correctness at min and max dynamic shape boundaries
- feat: capture JIT inspection details and config in tuning data reports - full inspection for `deferred` mode and basic for the `eager` mode
- feat: defer JIT deferred-mode tuning until the next normal forward pass after `aitune.torch.jit.tune.deferred()`
- breaking change: treat calls that pass the same forward parameters positionally or by keyword as the same graph; existing tuned checkpoints must be regenerated
- breaking change: use forward parameter paths instead of `args_*` and `kwargs_*` names in metadata reports and TensorRT optimization profiles; custom profiles must use the new paths
- breaking change: `aitune.torch.jit.tune.deferred()` now marks tuning to run on the next forward pass instead of tuning immediately
- breaking change: replace `FindMaxBatchSizeMixinConfig` and `PerformanceValidationMixinConfig` with a single strategy-level `ProfilingConfig`

## 0.4.1
- fix: use the recorded global batch size for baseline performance profiling
- fix: align Torch-TensorRT defaults and JIT inspection CUDA synchronization
- fix: remove memory spike during measuring performance baseline
- fix: remove memory spike for torch inductor jit backend with autocast
- fix: default profiling measurements to fixed-step sampling after warmup
- fix: compute profiling throughput from mean latency
- fix: stabilize optional CV-window profiling with warmup exclusion, max-sample bounds, and clearer failure errors
- fix: relax the default stable-window CV threshold to 10%
- fix: validate ratio and positive profiling configuration values
- breaking change: rename profiling and inspection threshold arguments to ratio-based names
- breaking change: replace `enable_validate_against_baseline(False)` with `enable_performance_validation(False)` as the single performance validation toggle

## 0.4.0
- feat: JIT tuning supports tune strategy selection via `aitune.torch.jit_config.strategy` (e.g. `MaxThroughputStrategy`); default remains `FirstWinsStrategy`
- feat: Add performance validation to tuning strategies. Backends with no speedup relative to TorchEagerBackend are skipped.
- feat: Deferred mode for Just-in-Time (JIT) tuning - explicit tuning trigger post whole pipeline or model pass, useful for image and video generation pipelines
- feat: Torch Inductor Ahead-of-Time (AoT) backend support added
- feat: ONNXRuntime backend support - CUDA and TensorRT execution providers
- feat: Extend TorchAO backend for NVFP4DQ and MXFP8DQ dynamic quantization options
- feat: Extend TensorRT backend for NVFP4 quantization options
- feat: Dynamo worker support - serve AITune-tuned models as Dynamo endpoints
- feat: New hardware metrics collection via subprocess-based hardware monitor
- feat: Hardware metrics output path can be configured via AITUNE_HARDWARE_METRICS_PATH environment variable
- feat: Locator now traverses `collections.UserDict` subclasses (e.g. `transformers.BatchEncoding`) like plain `dict`, restoring tensor shape tracking for sentence-transformers 5.4.0
- feat: JIT tuning can exclude packages or module classes from automatic tuning via `aitune.torch.jit_config.patch_exclude`
- feat: Add hardware metrics snapshot and enable/disable them at runtime
- feat: Tuning data collection emits a JSON report covering run, module, graph, and backend build stages - controlled via AITUNE_TUNING_DATA_COLLECTION environment variable
- feat: Tuning data output path can be configured via AITUNE_TUNING_DATA_PATH environment variable
- feat: Display speedup summary at WARNING level so it's visible at the default Python log level
- fix: Preserve externally-registered forward hooks (e.g. `capture_outputs` from transformers≥ 5) across the AOT and JIT save/restore cycle — hooks registered after AITune wraps a module are no longer lost on the first forward pass
- fix: Remove `transformers<5` restriction — AITune now supports transformers 5.x
- fix: Drop `enabled_precisions={float16}` default from Torch-TensorRT AOT and JIT backends; engine now matches the model's loaded dtype
- fix: Store checkpoint backend artifacts and SHA entries with relative paths so `.ait` archives remain portable after copy or move
- chore: Split hardware metrics table if there is a multi-gpu system
- chore: Renamed TorchInductorBackend to TorchInductorJitBackend - breaking change
- breaking change: NVTX_ENABLE environment variable renamed to AITUNE_NVTX_EVENTS
- breaking change: Removed SystemMonitor class from public API; use new hardware metrics collection instead
- breaking change: Removed system_resource_monitor function from public API
- breaking change: Removed enable_gpu_memory_logging function from public API
- breaking change: Removed `aitune.torch.jit_config.backends` — set `config.strategy = FirstWinsStrategy(backends=[...])` to customize backends


## 0.3.0
- feat: JIT tuning requires single sample only - tune on first model call
- feat: JIT tuning default fallback to Torch Inductor backend
- feat: allow to override tuning device in JIT tuning, when set to none, use the module device
- feat: add support for multi-profile engine with auto generated and user provided profiles in TensorRT backend
- fix: handling dynamic shapes in TorchTensorRT AoT backend
- fix: creating calibration data for ModelOpt ONNX PTQ in TensorRT backend
- misc: added documentation


## 0.2.0

- feat: introduce Just-in-Time (JIT) tuning: no-code model tuning controlled through import or environment flag
- feat: introduce Just-in-Time (JIT) inspect: no-code model analysis controlled through import or environment flag
- feat: module inspect considers lists and dicts for Torch module containers
- feat: add support for forward hooks for AOT and JIT tuning
- feat: add support for CUDA graphs for TensorRT backend
- feat: changing default ONNX export path to dynamo (torch.onnx.export(dynamo=True))
- feat: add ONNX AutoCast in TensorRT Backend for mixed precision through TensorRT ModelOpt
- feat: extend collecting profiling metrics through nvtx annotations
- feat: suppress console output during tuning and save logs to file - controlled through verbose flag
- feat: add support for dataclasses and user custom object in module.forward arguments
- feat: add support for kv cache for LLMs
- feat: add support for Static/Dynamic HuggingFace for TorchInductor backend
- feat: optimize handling input/output metadata
- feat: reduce CPU/GPU memory usage during tuning offloading modules to meta
- fix: prevent cache dir override when there are two similar modules in JIT tuning
- fix: dynamic shapes configuration for ONNX Dynamo export path in TensorRT Backend
- fix: bfloat16 support in TensorRT Backend
- fix: profiling of models without batching supported
- misc: extends examples and improved dependencies


## 0.1.0

- feat: add AITune features scoped for the first release
- feat: introduce Ahead-of-Time tuning for low-code model inspection and tuning
