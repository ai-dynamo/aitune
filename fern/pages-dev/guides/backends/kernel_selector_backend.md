---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Kernel Selector Backend Guide"
---

<Note>
The Kernel Selector backend is an experimental feature. Its APIs, supported providers, delegate compatibility, and
serialized plan format may change in future releases.
</Note>

`KernelSelectorBackend` is a composite backend. It selects optimized implementations for expensive
`torch.nn.functional` calls, applies the resulting `KernelOptimizationPlan`, and builds another AITune backend while
the selected providers are active. Inference is then delegated to that backend.

Use this backend when you want AITune tuning and checkpointing to manage provider selection. To select and apply
providers without compiling a module, use [`KernelOptimizer` directly](../advanced/kernel_provider.md).

## Quick start

By default, the backend evaluates PyTorch Flash Attention, PyTorch cuDNN Attention, and FlashAttention-4 for scaled
dot-product attention, then builds the optimized module with Torch Inductor JIT:

```python
import torch
import torch.nn.functional as F

from aitune.torch import Module
from aitune.torch.backend import KernelSelectorBackend
from aitune.torch.tune_strategy import OneBackendStrategy


class Attention(torch.nn.Module):
    def forward(self, query, key, value):
        return F.scaled_dot_product_attention(query, key, value)


backend = KernelSelectorBackend()
strategy = OneBackendStrategy(backend)
model = Module(Attention().eval().cuda(), "attention", strategy=strategy)

sample = tuple(
    torch.randn(2, 8, 256, 64, device="cuda", dtype=torch.float16)
    for _ in range(3)
)
model(*sample)  # Record representative inputs.
model.tune(device=torch.device("cuda"))
output = model(*sample)
```

CUDA is required for profiling, candidate validation, benchmarking, and GPU delegate builds. FlashAttention-4 is
evaluated when its optional runtime package is installed; an unavailable or unsupported provider does not prevent the
remaining candidates from being evaluated.

## Configuration

`KernelSelectorBackendConfig` accepts static providers, asynchronous generators, or both. When the configuration is
omitted, the default static attention providers are used. An explicitly configured backend requires at least one
provider or generator.

| Option | Default | Description |
|---|---:|---|
| `kernel_providers` | PyTorch Flash, PyTorch cuDNN, FlashAttention-4 | One `KernelProvider` or a list of static providers to evaluate. |
| `kernel_generators` | `None` | One `KernelGenerator` or a list of asynchronous generators to submit. |
| `provider_min_time_share_percent` | `0.0` | Minimum percentage of profiled CUDA time required before evaluating static providers for a function. |
| `generator_min_time_share_percent` | `10.0` | Minimum percentage of profiled CUDA time required before submitting generators for a function. |
| `generation_timeout` | `21600.0` | Maximum time, in seconds, to wait for generated candidates. The default comes from `AITUNE_KERNEL_GENERATION_TIMEOUT`. |

`KernelSelectorBackend()` uses `TorchInductorJitBackend()` as its default delegate. Pass `config` or
`delegate_backend` explicitly to customize either part of the composite backend.

The optimizer validates candidate outputs and benchmarks valid candidates against the original PyTorch function. A
provider enters the plan only when it supports all representative samples, passes correctness validation, and is
faster than the original function.

An empty plan is valid. In that case, the delegate builds the unchanged module.

## Delegate behavior

The Kernel Selector backend is not a compiler. Its build mode and supported execution modes are inherited from
`delegate_backend`.

During `tune()`, AITune:

1. profiles the module and creates a `KernelOptimizationPlan`;
2. temporarily applies the selected providers;
3. builds the delegate while those providers are active;
4. retains or discards the live provider runtime according to the delegate build mode.

### JIT delegates

For a JIT delegate, the selected plan remains active around inference because compilation may happen lazily or recur
for new input specializations. A checkpoint stores the selected plan and the delegate state. Loading the checkpoint
restores the plan without profiling or selecting providers again.

### AOT delegates

For an AOT delegate, provider calls must be captured into the compiled delegate artifact during the build. The live
provider runtime is then discarded. A checkpoint stores the delegate artifact and does not reinstall provider hooks.

Provider compatibility therefore depends on the delegate's ability to capture, partition, compile, and serialize the
selected implementation. A provider that works with one delegate is not automatically compatible with every other
delegate.

## Checkpoints

Use the regular AITune checkpoint APIs:

```python
from aitune.torch import load, save

save(model, "attention.ait")
restored = load(
    Attention().eval(),
    "attention.ait",
    device_map={"": torch.device("cuda")},
)
```

For JIT delegates, the restored backend activates the serialized provider plan before deploying the delegate. For AOT
delegates, deployment loads the compiled delegate artifact.

## SageAttention V1 and Torch-TensorRT

The `sageattention` package from PyPI provides SageAttention V1, whose attention implementation uses Triton. The
following delegate behavior has been validated:

- Torch Inductor JIT and AOT can capture and compile the Triton implementation.
- Torch-TensorRT JIT can leave the unsupported Triton attention path in PyTorch while compiling supported neighboring
  operations into TensorRT partitions.
- Torch-TensorRT AOT is not supported with SageAttention V1 because its export and artifact serialization path cannot
  reliably package the captured Triton implementation.

For a model containing SageAttention followed by one linear projection, only the projection is eligible for TensorRT.
Set `min_block_size=1` so Torch-TensorRT compiles that single-operation partition instead of skipping it under its
default minimum partition size:

```python
from aitune.torch.backend import (
    TorchTensorRTConfig,
    TorchTensorRTJitBackend,
    TorchTensorRTJitBackendConfig,
)

delegate = TorchTensorRTJitBackend(
    TorchTensorRTJitBackendConfig(
        compile_config=TorchTensorRTConfig(min_block_size=1),
    )
)
```

This configuration does not run the SageAttention Triton operation inside TensorRT. It only makes the neighboring
supported partition eligible for TensorRT conversion.

See [Kernel Providers](../advanced/kernel_provider.md) for built-in providers, direct optimizer use, provider lifecycle,
and custom provider or generator contracts.
