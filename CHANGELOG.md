---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Changelog"
---
# Changelog

## 0.4.0 (unreleased)
- feat: JIT patcher exclusions (`extra_patch_exclude_packages`, `extra_patch_exclude_modules`) are now configurable on `jit_config`
- feat: JIT tuning supports tune strategy selection via `aitune.torch.jit_config.strategy` (e.g. `MaxThroughputStrategy`); default remains `FirstWinsStrategy`
- feat: Add performance validation to tuning strategies. Backends with no speedup relative to TorchEagerBackend are skipped.
- feat: Deferred mode for Just-in-Time (JIT) tuning - explicit tuning trigger post whole pipeline or model pass, useful for image and video generation pipelines
- feat: Torch Inductor Ahead-of-Time (AoT) backend support added
- feat: ONNXRuntime backend support - CUDA and TensorRT execution providers
- feat: Extend TorchAO backend for NVFP4DQ and MXFP8DQ dynamic quantization support
- feat: Dynamo worker support - serve AITune-tuned models as Dynamo endpoints
- feat: New hardware metrics collection via subprocess-based hardware monitor
- feat: Hardware metrics output path can be configured via AITUNE_HARDWARE_METRICS_PATH environment variable
- feat: Locator now traverses `collections.UserDict` subclasses (e.g. `transformers.BatchEncoding`) like plain `dict`, restoring tensor shape tracking for sentence-transformers 5.4.0
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
