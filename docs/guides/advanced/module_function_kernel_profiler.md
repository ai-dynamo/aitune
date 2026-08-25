---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Module Function Kernel Profiler"
---

`ModuleFunctionKernelProfiler` shows which CUDA kernels are launched by
`torch.nn.functional` calls inside a module hierarchy. It attributes every recorded kernel to:

- the nearest profiled module;
- the `torch.nn.functional` call;
- the underlying PyTorch operator;
- the CUDA kernel and its duration.

It can also collect distinct input samples and their call counts for selected functional calls. This is useful when
identifying frequently executed operations, preparing representative kernel benchmarks, or deciding which functions
are worth optimizing.

See [Kernel Providers](kernel_provider.md) to prepare, validate, benchmark, and activate alternative implementations for
the functional calls identified by the profiler.

This profiler answers a different question from [`aitune.torch.profile()`](performance_profile.md):

| API | Use it to answer |
|---|---|
| `aitune.torch.profile()` | Where is end-to-end inference time spent across modules and pipeline code? |
| `ModuleFunctionKernelProfiler` | Which module-scoped functional calls launch each CUDA kernel, and with which inputs? |

## Quick start

When the inference callable is an `nn.Module`, the profiler infers the module hierarchy automatically:

```python
import torch
from torch import nn

import aitune.torch as ait

model = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
).eval().cuda()

samples = [
    ((torch.randn(32, 128, device="cuda"),), {}),
]

profiler = ait.ModuleFunctionKernelProfiler(
    function_names={"linear", "relu"},
)
profiling_df, function_data = profiler.profile(
    model,
    samples,
    warmup_iterations=3,
)

summary_df = profiler.describe_results(profiling_df, function_data)
print(summary_df)
```

Each item in `samples` is an `(args, kwargs)` pair. A warmup iteration executes the complete list once. After warmup,
the profiler executes the complete list once more while recording events.

## Profiling a module used by a pipeline

The inference callable does not need to be the module itself. Pass `module` when a pipeline or another function invokes
the module you want to observe:

```python
profiler = ait.ModuleFunctionKernelProfiler()


def inference_fn():
    pipe("A futuristic cityscape", num_inference_steps=1)


profiling_df, function_data = profiler.profile(
    inference_fn,
    module=pipe.transformer,
)
```

When `data` is omitted, each warmup iteration and the recorded iteration call `inference_fn()` without arguments.

## Profiling results

`profiling_df` contains one row per attributed CUDA kernel:

| Column | Meaning |
|---|---|
| `module_name` | Name from the profiled module hierarchy. The root module uses an empty name. |
| `module` | The corresponding live `nn.Module` object. |
| `function_name` | Name of the intercepted `torch.nn.functional` call, such as `linear` or `relu`. |
| `op` | PyTorch profiler operator, such as `aten::linear`. |
| `kernel` | CUDA kernel name reported by PyTorch Profiler. |
| `kernel_us` | Kernel duration in microseconds. |

`describe_results()` groups the most expensive functional calls (up to `top_k`, default 10) and returns one row per
function:

| Column | Meaning |
|---|---|
| `function_name` | Name of the intercepted `torch.nn.functional` call, such as `linear` or `relu`. |
| `calls` | Number of times the function was invoked. `NaN` if sample collection was disabled for this function. |
| `num_distinct_samples` | Number of distinct input samples. `NaN` if sample collection was disabled. |
| `num_modules` | Number of distinct profiled modules that invoked this function. |
| `time_spent_us` | Total CUDA kernel time attributed to this function, in microseconds. |
| `time_spent_pct` | Share of kernel time across all profiled functions, including functions omitted by `top_k`. |
| `tensor_size_MB` | Total size of input tensors across distinct samples, in megabytes. `NaN` if sample collection was disabled. |

Example dataframe:
```text
                  function_name  calls  num_distinct_samples  num_modules  time_spent_us  time_spent_pct  tensor_size_MB
0  scaled_dot_product_attention      1                     1            1       2201.655       84.823238        0.750000
1                    group_norm     10                     2           10        251.011        9.670709        0.500061
2                        linear      4                     1            4         79.199        3.051303        0.250076
3                        conv2d     10                     3           10         51.747        1.993658        0.563641
4                          silu      9                     1            5         11.968        0.461092        0.250000
```

Note: `time_spent` is calculated for that particular function excluding calls for inner functions e.g. for a `multi_head_attention_forward` which calls `linear` and `scaled_dot_product_attention`, `time_spent` will cover each function own time.

## Collected input samples

`function_data` maps each collected function name to `(call_count, sample)` pairs. Each sample has the same
`(args, kwargs)` form accepted by `profile()`:

```python
for call_count, (args, kwargs) in function_data["linear"]:
    print(call_count, args, kwargs)
```

`function_names` controls only sample collection; it does not filter rows from `profiling_df`. Use:

- `None` to collect inputs for every observed functional call;
- a set such as `{"linear", "conv2d"}` to collect inputs only for those calls;
- an empty set to collect no inputs while retaining kernel profiling.

Input samples retain references to their tensors until the next `profile()` call or until the profiler is released.
Restrict `function_names` when profiling workloads with many distinct inputs to reduce retained memory.

## Runtime behavior and requirements

- CUDA is required.
- Warmup and recorded inference run under `torch.no_grad()`.
- Module forwards and supported `torch.nn.functional` functions are instrumented only for the duration of `profile()`
  and restored afterward, including when inference raises an exception.
- Do not concurrently execute inference that uses the same modules or functional namespace while profiling.
