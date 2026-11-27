---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Backend Catalog"
---

Use this catalog to find the backend guide for your model or deployment target. A backend builds and runs a
candidate implementation. The tuning strategy decides which candidates to evaluate and which one to use.
See [Core Concepts](../../learn/concepts.md#backends-and-strategies) and
[Tune Strategies](../tune_strategies/tune_strategies.md).

| Backend | What it runs |
| --- | --- |
| [TensorRT](tensorrt_backend.md) | Builds and runs a TensorRT engine from a PyTorch module or existing ONNX model. |
| [Torch-TensorRT JIT](torch_tensorrt_jit_backend.md) | Integrates TensorRT through PyTorch's `torch.compile`. |
| [Torch-TensorRT AOT](torch_tensorrt_aot_backend.md) | Compiles with Torch-TensorRT ahead of backend execution. |
| [Torch Inductor JIT](torch_inductor_jit_backend.md) | Compiles through PyTorch's `torch.compile` with Inductor. |
| [Torch Inductor AOT](torch_inductor_aot_backend.md) | Builds a compiled Inductor artifact for reuse. |
| [TorchAO](torchao_backend.md) | Applies TorchAO quantization to PyTorch modules. |
| [ONNX Runtime](onnx_runtime_backend.md) | Runs an exported PyTorch module or existing ONNX model with an ONNX Runtime execution provider. |
| [Kernel Selector](kernel_selector_backend.md) | Evaluates kernel implementations for supported PyTorch functions before building through a delegate backend. This backend is experimental. |

For an existing ONNX file, use the [guide to tuning an existing ONNX model](../onnx_tuning.md) to choose between TensorRT and
ONNX Runtime. To provide your own backend, see [Extending AITune](../advanced/extending_aitune.md).
