---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Known Issues and Limitations"
---

- **Diffusers context parallelism with JIT tuning.** AITune's automatic JIT tuning currently degrades inference
  performance with Diffusers context parallelism because of module wrapping and patching order. Use the explicit AOT
  tuning API as a workaround. JIT backends may still be selected through the AOT API.
- **Transformers 5.16 or later native tensor parallelism with JIT tuning.** Transformers 5.16 replaced its legacy
  tensor-parallel implementation with a DTensor-native backend. AITune's automatic JIT tuning is not currently
  compatible with the DTensor forward hooks and can fail with mixed `Tensor` and `DTensor` inputs. Use the explicit
  AOT [`inspect`](guides/aot_inspect.md), [`wrap`](guides/aot_tuning.md), and [`tune`](guides/aot_tuning.md) flow
  instead. JIT backends may still be selected through the AOT API.
- **Symmetric distributed tuning.** AOT and JIT tuning require every rank to process the same tuning candidates and
  enter collective build, validation, and profiling operations in the same order. AOT tuning must be invoked on every
  rank with an equivalent tuning plan. JIT tuning additionally requires every rank to reach tuning readiness on the
  same forward. Divergent execution can fail or deadlock.
- **Distributed inspection filtering.** Unfiltered `InspectedModulesInfo.get_modules()` results preserve discovery
  order so that symmetric models produce a stable candidate order across ranks. Passing `limit` or
  `min_execution_ratio` ranks candidates by locally measured execution time. Timing differences between ranks can
  therefore select or order candidates differently and cause AOT tuning-plan validation to fail. For distributed
  tuning with either filter, select the module paths on one rank and broadcast that selection to every rank before
  wrapping the modules.
- **Distributed checkpoints.** Multi-GPU tuning produces rank-local artifacts. Loading with a different world size or
  rank topology is not supported, and AITune does not currently package distributed artifacts into one portable
  checkpoint.
- **Kernel providers with strict Torch Export on PyTorch 2.9.** PyTorch 2.9 strict export cannot capture the forward
  hooks that temporarily replace `torch.nn.functional` kernels and can fail with a guard-export assertion. This is a
  generic limitation affecting every kernel provider, including SageAttention, when `KernelOptimizerBackend` uses a
  `TorchInductorAotBackend` delegate. Use PyTorch 2.10 or later, or on PyTorch 2.9 choose a delegate that does not rely
  on strict Torch Export, such as a JIT delegate.
- **SageAttention V1 with a Torch-TensorRT AOT delegate.** The SageAttention package on PyPI provides a Triton-based V1
  implementation that the Torch-TensorRT AOT export and serialization path cannot reliably package. When using
  `KernelOptimizerBackend` with `SageAttentionKernelProvider`, use a Torch Inductor JIT delegate, a Torch Inductor AOT
  delegate with PyTorch 2.10 or later, or a Torch-TensorRT JIT delegate when partial compilation with PyTorch fallback
  is acceptable. See the
  [Kernel Optimizer Backend Guide](guides/backends/kernel_optimizer_backend.md#sageattention-v1-and-torch-tensorrt).
