---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Kernel Providers"
---

<Note>
Kernel providers are an experimental feature. Their APIs, supported implementations, runtime behavior, and serialized
plan format may change in future releases.

</Note>

`KernelOptimizer` finds expensive `torch.nn.functional` calls inside a module hierarchy, evaluates compatible kernel
providers, and returns a plan containing only candidates that are faster than the original PyTorch functions. The plan
can be activated directly, without wrapping the module in an AITune backend.

Use the direct optimizer when you want to:

- optimize individual functional calls without compiling the complete module;
- control which kernel implementations participate in selection;
- optimize a module invoked from a larger model or pipeline;
- inspect, serialize, or activate the selected provider plan yourself.

CUDA is required for profiling and benchmarking.

Provider preparation, correctness validation, and benchmarking run under `torch.no_grad()`. Applying a plan only
manages provider activation, so inference code should explicitly use `torch.no_grad()`.

## How kernel providers work

The optimization flow has two separate phases:

1. `KernelOptimizer.make_plan()` profiles the workload, prepares, validates, and benchmarks candidates, and returns an
   immutable `KernelOptimizationPlan`. It does not modify the module.
2. `plan.apply(module)` installs forward hooks on the selected module for the duration of a context. During its forward
   pass, matching `torch.nn.functional` calls are temporarily redirected to the selected providers.

The optimizer:

1. profiles the supplied inference callable and collects representative inputs for functions supported by configured
   providers or generators;
2. ranks all observed functional calls by CUDA kernel time and keeps the global `top_k`;
3. submits eligible asynchronous generators before evaluating static providers;
4. calls `prepare()` on compatible static providers to derive the state required for inference;
5. validates every prepared or generated provider against the original PyTorch function under `torch.no_grad()`;
6. benchmarks valid candidates under `torch.no_grad()` using the observed input distribution;
7. selects the fastest provider for each function only when it beats the original function.

An unavailable optional runtime or a failing candidate is isolated and skipped while the remaining candidates continue.
SageAttention and FlashAttention-4 load their runtime functions lazily during provider inference; `prepare()` only derives
an inference plan from the representative samples.

Because `top_k` is applied before filtering for functions supported by the configured providers and generators, increase
it when the target function is not among the most expensive calls in the workload.

## Direct optimizer example

The following example evaluates the PyTorch SDPA implementations available on the current GPU. It uses
`KernelOptimizer` directly and does not create an AITune backend:

```python
import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend

from aitune.torch.backend.kernels import KernelOptimizer
from aitune.torch.backend.kernels.kernel_provider import TorchSDPAKernelProvider

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)


class Attention(nn.Module):
    def forward(self, query, key, value):
        return F.scaled_dot_product_attention(query, key, value, is_causal=True)


model = Attention().eval().cuda()
sample = tuple(
    torch.randn(2, 8, 256, 64, device="cuda", dtype=torch.float16)
    for _ in range(3)
)
data = [(sample, {})]

optimizer = KernelOptimizer(
    top_k=5,
    kernel_providers=[
        TorchSDPAKernelProvider(SDPBackend.MATH),
        TorchSDPAKernelProvider(SDPBackend.EFFICIENT_ATTENTION),
        TorchSDPAKernelProvider(SDPBackend.CUDNN_ATTENTION),
        TorchSDPAKernelProvider(SDPBackend.FLASH_ATTENTION),
    ],
)
plan = optimizer.make_plan(model, data)

with plan.apply(model):
    output = model(*sample)
```

A plan can be empty when no candidate supports every representative input, passes correctness validation, or improves
on the baseline. An empty plan is a valid result: `plan.apply(model)` leaves the module running its original PyTorch
functions.

## Built-in providers

AITune includes the following static providers:

| Provider | Runtime dependency | Purpose |
|---|---|---|
| `TorchSDPAKernelProvider` | PyTorch | Runs SDPA under a selected `torch.nn.attention.SDPBackend`. |
| `SageAttentionKernelProvider` | `sageattention` | Runs compatible SDPA calls with SageAttention. |
| `FlashAttention4KernelProvider` | `flash-attn-4` | Runs compatible 4D SDPA calls with FlashAttention-4, including supported GQA and MQA layouts. |

FlashAttention-4 targets Hopper and Blackwell GPUs, such as H100 and B200. The PyPI `sageattention` package provides
SageAttention V1; for newer implementations, install SageAttention from its source repository.

Optional providers can be configured together. An unavailable or incompatible implementation is skipped while the
remaining candidates continue through validation and benchmarking:

```python
from torch.nn.attention import SDPBackend

from aitune.torch.backend.kernels import KernelOptimizer
from aitune.torch.backend.kernels.kernel_provider import (
    FlashAttention4KernelProvider,
    SageAttentionKernelProvider,
    TorchSDPAKernelProvider,
)

optimizer = KernelOptimizer(
    kernel_providers=[
        FlashAttention4KernelProvider(),
        SageAttentionKernelProvider(),
        TorchSDPAKernelProvider(SDPBackend.FLASH_ATTENTION),
    ],
)
```

Providers specialize their inference plans from representative samples. Inconsistent or unsupported sample plans can
cause `prepare()` to return `False`; runtime and correctness failures reject the candidate before it can enter the
selected plan.

## Provider interface and lifecycle

Import the base provider APIs from `aitune.torch.backend.kernels.kernel_provider`:

```python
from aitune.torch.backend.kernels.kernel_provider import (
    KernelProvider,
    KernelProviderState,
    kernel_provider_from_dict,
)
```

A provider implements one `torch.nn.functional` function and has two states:

| State | Meaning |
|---|---|
| `KernelProviderState.INIT` | The provider has not yet accepted representative samples. |
| `KernelProviderState.READY` | The provider is callable and serializable. |

Subclasses implement:

| Member | Contract |
|---|---|
| `supported_function` | Abstract property containing the name of one `torch.nn.functional` function. |
| `_prepare(samples)` | Validate representative samples, populate inference state, and return `True`; return `False` for unsupported samples. |
| `_infer(*args, **kwargs)` | Execute the prepared implementation. |
| `_to_dict()` | Serialize provider-specific inference state. |
| `_from_dict(state_dict)` | Restore provider-specific inference state. |

The public `prepare(samples)` method is idempotent. A successful call changes the state from `INIT` to `READY`; repeated
calls on a ready provider return `True` without rebuilding its state. Calling the provider or `to_dict()` before it is
ready raises `RuntimeError`.

The default `name` and `repr(provider)` use the provider class name. Providers may override `name` with a more useful
description. Each subclass is registered under its class name when its module is imported, allowing
`kernel_provider_from_dict()` and `KernelOptimizationPlan.from_dict()` to restore the concrete class. Import custom
provider classes before restoring plans that contain them.

## Asynchronous kernel generators

Kernel generators produce providers asynchronously and are exported from the same package:

```python
from aitune.torch.backend.kernels.kernel_provider import (
    KernelGenerationResult,
    KernelGenerator,
)
```

A `KernelGenerator` implements:

- `__repr__()` for a human-readable description;
- `supports_functions()` to list supported `torch.nn.functional` function names;
- `prepare(function, samples)` to determine whether generation can handle all samples;
- `submit(function, samples)` to return a `Future[KernelGenerationResult]` without waiting for generation to finish.

A `KernelGenerationResult` contains the function name, description, and exactly one of a generated `provider` or an
`error` message. A successful generator must return a ready, serializable provider because the optimizer immediately
uses it for correctness validation and benchmarking. Generator exceptions are isolated, and unfinished futures are
cancelled or ignored after `generation_timeout`.

## Optimizing a module inside a pipeline

The inference callable and optimized module can be different objects. This is useful when a pipeline prepares inputs
or invokes the target module internally:

```python
prompt = "A futuristic cityscape"
data = [((prompt,), {})]

plan = optimizer.make_plan(
    pipe,
    data,
    module=pipe.transformer,
)
with plan.apply(pipe.transformer):
    result = pipe(prompt)
```

`function` is the callable executed for profiling. `module` defines the module hierarchy in which functional calls are
attributed and later redirected to providers. When `function` is an `nn.Module`, it is also used as `module` by default.

See [Module Function Kernel Profiler](module_function_kernel_profiler.md) for details about function attribution and
representative input collection.

## Runtime lifecycle

For temporary activation, apply the plan directly. Hooks are removed when the context exits, including when inference
raises an exception:

```python
with plan.apply(model):
    output = model(*sample)
```

`plan.apply()` enters `torch.no_grad()` for inference and restores the previous gradient state when the context exits.

Use `KernelProviderRuntime` directly when activation must span multiple contexts or requires explicit lifecycle control:

```python
from aitune.torch.backend.kernels import KernelProviderRuntime

runtime = KernelProviderRuntime(model, plan)
with torch.no_grad():
    runtime.activate()
    try:
        output = model(*sample)
    finally:
        runtime.deactivate()

with runtime.applied():
    output = model(*sample)
```

Activation and deactivation are idempotent. If a runtime is already active, `applied()` preserves that state when the
context exits. Direct `activate()` and `deactivate()` calls only manage provider hooks, so callers using that lifecycle
must continue to manage the inference context explicitly.

The runtime temporarily changes process-global `torch.nn.functional` attributes during the selected module's forward
pass. Do not run concurrent forwards that overlap an active provider scope in different threads.

## Saving and restoring a plan

`KernelOptimizationPlan.providers` is an immutable tuple of prepared providers. Plans serialize it under the
`"providers"` key and can be restored without profiling again:

```python
import json
from pathlib import Path

import torch

from aitune.torch.backend.kernels import KernelOptimizationPlan

plan_path = Path("kernel-plan.json")
plan_path.write_text(json.dumps(plan.to_dict()))

restored_plan = KernelOptimizationPlan.from_dict(
    json.loads(plan_path.read_text())
)
with restored_plan.apply(model):
    output = model(*sample)
```

Each serialized provider includes a `"type"` field derived from its class name and its provider-specific inference
state. Deserialization restores providers directly in the `READY` state. Optional runtime packages required by selected
providers must be installed when optimized inference runs. A restored plan remains specialized to the functional call
patterns represented by the samples used during optimization.

## Configuration

The most relevant `KernelOptimizer` options are:

| Option | Default | Meaning |
|---|---:|---|
| `top_k` | `5` | Number of globally most expensive functional calls considered for optimization. |
| `kernel_providers` | `[]` | Static providers to prepare, validate, and benchmark. A single provider is accepted. |
| `provider_min_time_share_percent` | `0.0` | Minimum share of total profiled kernel time required before evaluating static providers for a function. |
| `kernel_generators` | `[]` | Asynchronous kernel generators evaluated alongside static providers. A single generator is accepted. |
| `generator_min_time_share_percent` | `10.0` | Minimum profiled time share required before submitting generators for a function. |
| `generation_timeout` | `AITUNE_KERNEL_GENERATION_TIMEOUT` or `21600` | Maximum time in seconds to wait for submitted generator futures. |

Representative `data` uses the same `[(args, kwargs), ...]` structure as
`ModuleFunctionKernelProfiler.profile()`. Include every input shape, dtype, layout, and argument combination that the
applied plan is expected to handle.
