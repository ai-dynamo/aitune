<!--
Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Changelog

## 0.3.0 (unreleased)
- feat: allow to override tuning device in JIT tuning, when set to none, use the module device
- fix: handling dynamic shapes in TorchTensorRT AoT backend


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
