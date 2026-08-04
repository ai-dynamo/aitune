---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Tune Strategies Guide"
---

Tune strategies determine how AITune selects and configures backends during the tuning process. They provide flexibility in balancing performance, reliability, and tuning time.

## Overview

AITune provides five built-in strategies:

- **OneBackendStrategy**: Uses a single specified backend
- **FirstWinsStrategy**: Tries backends in order, uses the first that succeeds
- **MaxThroughputStrategy**: Profiles all backends, selects the one with the highest throughput
- **MinLatencyStrategy**: Profiles all backends, selects the one with the lowest latency
- **LatencyBudgetStrategy**: Profiles all backends, selects the highest throughput result that stays within a latency budget

## Why Backends Can Fail

Not every backend can successfully tune every model. Each backend relies on a different compilation or export technology, and each has its own limitations:

- **TensorRT** requires exporting the model to ONNX. Models with unsupported operators, complex dynamic control flow, or symbolic shape constraints may fail during ONNX export or TensorRT engine building. Memory constraints can also prevent the engine from being built.
- **Torch Inductor** uses `torch.compile`, which may encounter *graph breaks* on unsupported Python constructs or operations, causing partial or failed compilation.
- **TorchAO** applies quantization transformations that may not support all layer types or model architectures.
- **Torch-TensorRT** combines PyTorch's compiler with TensorRT, inheriting potential limitations from both.

Because of these differences, a backend that fails on one model may succeed on another, and vice versa. This is the core motivation behind strategies like `FirstWinsStrategy`: by trying multiple backends in priority order, you get automatic fallback when your preferred backend cannot handle a particular model.

## Performance Validation

Strategies validate both correctness and performance before accepting a tuned backend. When performance validation is enabled, AITune profiles a `TorchEagerBackend` baseline at the resolved batch size, then profiles each correctness-passing backend against that baseline.

For `OneBackendStrategy` and `FirstWinsStrategy`, baseline validation is enabled by default. A backend is rejected when its throughput is below `1 + min_speedup_ratio` relative to Torch eager; the default threshold is 1%, so a backend must be at least 1.01x faster to pass. `FirstWinsStrategy` then tries the next backend. `OneBackendStrategy` falls back to the profiled `TorchEagerBackend` when its single backend is correct but not faster. Disable this only when you deliberately want to keep a backend that is correct but not faster:

```python
strategy.enable_performance_validation(False)
```

`MaxThroughputStrategy` also profiles Torch eager as a baseline. With performance validation enabled, it falls back to Torch eager if no user-provided backend beats the baseline. When disabled with `enable_performance_validation(False)`, the Torch eager baseline is skipped and the fastest successful user backend wins.

`MinLatencyStrategy` works the same way but selects by minimum latency instead of maximum throughput. `LatencyBudgetStrategy` selects by maximum throughput after filtering out profiled batch sizes whose latency exceeds the configured budget. Use `enable_performance_validation(False)` to disable baseline comparison.

## Choosing a Strategy

Use the table below as a quick decision guide. If you already know a backend is compatible and stable in production, start with `OneBackendStrategy`. If you want a safer default with minimal tuning time, `FirstWinsStrategy` balances reliability and speed. When absolute throughput matters and you can afford longer tuning, choose `MaxThroughputStrategy`.

| Strategy                  | When to Use                                   | Tuning Time | Reliability             | Performance                     |
|---------------------------|-----------------------------------------------|-------------|-------------------------|---------------------------------|
| OneBackendStrategy        | Known backend works, production               | Fast        | High (if backend works) | Depends on backend              |
| FirstWinsStrategy         | Want reliability, quick tuning                | Fast-Medium | Very High               | Good                            |
| MaxThroughputStrategy     | Maximum throughput, batch workloads           | Slow        | High                    | Best throughput                 |
| MinLatencyStrategy        | Minimum latency, fixed batch size workloads   | Slow        | High                    | Best latency at given batch size|
| LatencyBudgetStrategy     | Maximum throughput with a latency SLO         | Slow        | High                    | Best throughput under budget    |

## OneBackendStrategy

Uses exactly one backend, failing immediately with the original error if it cannot build or validate correctness. If the backend is correct but does not pass the eager performance gate, the strategy falls back to `TorchEagerBackend`. Use this when you have already validated that a backend works and want deterministic, reproducible behavior in production.

<Note>
`OneBackendStrategy` may look equivalent to `FirstWinsStrategy` with a single backend, but the key difference is error handling: `OneBackendStrategy` raises the backend's original build and correctness exceptions, while `FirstWinsStrategy` catches errors and tries the next candidate.

</Note>
### Usage

```python
from aitune.torch.backend import TensorRTBackend, TensorRTBackendConfig
import aitune.torch as ait

# Configure backend
config = TensorRTBackendConfig()
backend = TensorRTBackend(config)

# Create strategy
strategy = ait.OneBackendStrategy(backend=backend)

# Use in tuning
model = ait.Module(model, "my-model", strategy=strategy)
ait.tune(model, input_data)
```

### When to Use

✅ **Good for**:

- Production environments with validated backends
- Reproducible results
- Fast tuning cycles
- Specific backend requirements

❌ **Not ideal for**:

- Experimentation (no fallback)
- Unknown models (might fail)
- Maximum performance discovery

## FirstWinsStrategy

Tries backends in priority order and returns the first one that successfully builds, passes correctness checks, and meets the performance threshold. If a backend fails or is slower than the Torch eager baseline, the strategy moves on to the next candidate instead of aborting. If all backends fail, the error is caught and the original model is used as-is. List backends from fastest to most compatible — for example, TensorRT first, then Torch Inductor.

### Usage

```python
from aitune.torch.backend import (
    TensorRTBackend,
    TensorRTBackendConfig,
    TorchInductorJitBackend,
)
import aitune.torch as ait

# List backends in priority order (fastest → most compatible)
backends = [
    TensorRTBackend(config=TensorRTBackendConfig()),  # Best performance, but may not support all models
    TorchInductorJitBackend(),                            # Good performance, broader compatibility
]

# Create strategy
strategy = ait.FirstWinsStrategy(backends=backends)

# Use in tuning
model = ait.Module(model, "my-model", strategy=strategy)
ait.tune(model, input_data)
```

### How It Works

1. Tries first backend (e.g., TensorRT)
2. If successful → uses it, done
3. If fails or is slower than the Torch eager baseline → tries next backend
4. Repeats until a backend succeeds or all fail

### When to Use

✅ **Good for**:

- Experimentation with unknown or diverse models
- CI/CD pipelines where different models may need different backends
- Maximum reliability (if all backends fail, the original model is used as-is)
- Quick validation that *something* works before investing in backend-specific tuning

❌ **Not ideal for**:

- Maximum performance (stops at first success, not the fastest)
- When you already know which backend works (use `OneBackendStrategy` instead)
- Detailed performance comparison (use `MaxThroughputStrategy` instead)

### Best Practices

1. **Order by Performance**: Put fastest backends first
2. **Automatic Fallback**: If all backends fail, the original model is used as-is
3. **Similar Configurations**: Use compatible configs (e.g., all FP16)

## MaxThroughputStrategy

Tries all backends, profiles their performance, and selects the fastest backend that beats the Torch eager baseline.

### Usage

```python
from aitune.torch.backend import (
    TensorRTBackend,
    TensorRTBackendConfig,
    TorchInductorJitBackendConfig,
    TorchInductorJitBackend,
    TorchAOBackend,
    TorchAOBackendConfig
)
import aitune.torch as ait

# List all candidate backends
backends = [
    TensorRTBackend(config=TensorRTBackendConfig()),
    TorchInductorJitBackend(config=TorchInductorJitBackendConfig(mode="max-autotune")),
    TorchAOBackend(config=TorchAOBackendConfig(quantization="fp8wo")),
]

# Create strategy
strategy = ait.MaxThroughputStrategy(
    backends=backends,
)

# Use in tuning
model = ait.Module(model, "my-model", strategy=strategy)
ait.tune(model, input_data)
```

### How It Works

1. Tries to build with each backend
2. For successful backends:
   - Runs warmup iterations
   - Measures throughput over N iterations
   - Records performance metrics
3. Compares throughput with the Torch eager baseline
4. Selects the fastest user backend when it beats the baseline, otherwise falls back to Torch eager

### When to Use

✅ **Good for**:

- Production deployment planning
- Maximum performance requirements
- Comparing backend options
- When tuning time is not critical

❌ **Not ideal for**:

- Quick experiments (slow)
- Development iteration (overkill)
- Memory-constrained systems (keeps multiple builds)

## MinLatencyStrategy

Tries all backends, profiles their latency at batch size 1, and selects the backend with the lowest latency that beats the Torch eager baseline. Use this instead of `MaxThroughputStrategy` when you care about per-request response time rather than throughput.

### Usage

```python
from aitune.torch.backend import TensorRTBackend, TorchInductorJitBackend
from aitune.torch.task.profiling import (
    ModelExecutionTimeMeasuringStrategy,
    NumStepsMeasuringStopStrategy,
    ProfilingConfig,
)
from aitune.torch.task.profiling.profiling_stop_strategy import AllSamplesProfilingStopStrategy
import aitune.torch as ait

backends = [
    TensorRTBackend(),
    TorchInductorJitBackend(),
]

strategy = ait.MinLatencyStrategy(
    backends=backends,
    profiling_config=ProfilingConfig(
        batch_sizes=[1],
        measuring_strategy=ModelExecutionTimeMeasuringStrategy(),
        measurement_stop_strategy=NumStepsMeasuringStopStrategy(num_steps=50),
        profiling_stop_strategy=AllSamplesProfilingStopStrategy(),
    ),
)

model = ait.Module(model, "my-model", strategy=strategy)
ait.tune(model, input_data)
```

`MinLatencyStrategy` always profiles at batch size 1 regardless of `profiling_config.batch_sizes`.

### When to Use

✅ **Good for**:

- Real-time inference at batch size 1
- Latency-sensitive serving (e.g. per-request inference)
- Comparing backend options by response time

❌ **Not ideal for**:

- Batch workloads where throughput matters more (use `MaxThroughputStrategy`)

## LatencyBudgetStrategy

Tries all backends, profiles throughput and latency across the configured batch sizes, filters out results above the latency budget, and selects the backend/batch-size pair with the highest remaining throughput. The tuning report records `throughput`, `latency`, and `selected_batch_size` for each successful backend. If no user-provided backend satisfies the budget, tuning raises.

### Usage

```python
from aitune.torch.backend import TensorRTBackend, TorchInductorJitBackend
import aitune.torch as ait

strategy = ait.LatencyBudgetStrategy(
    latency_budget_ms=50.0,
    backends=[
        TensorRTBackend(),
        TorchInductorJitBackend(),
    ],
)

model = ait.Module(model, "my-model", strategy=strategy)
ait.tune(model, input_data, batch_sizes=[1, 2, 4, 8, 16])
```

### When to Use

✅ **Good for**:

- Services with a latency SLO
- Batch-size tuning where larger batches improve throughput but may exceed latency targets
- Selecting the fastest compliant backend for online inference

❌ **Not ideal for**:

- Strict minimum-latency workloads (use `MinLatencyStrategy`)
- Pure offline batch workloads with no latency constraint (use `MaxThroughputStrategy`)

## Best Practices

1. **Development**: Use `FirstWinsStrategy` with fallback
2. **Production**: Use `OneBackendStrategy` with validated backend
3. **Benchmarking**: Use `MaxThroughputStrategy` to find the best option, or `LatencyBudgetStrategy` when you have a latency SLO
4. **Always Validate**: Test tuned models before deployment
5. **Cache Results**: Save tuned models to avoid re-tuning

## Next Steps

- Learn about specific backends: [TensorRT](../backends/tensorrt_backend.md), [ONNXRuntime](../backends/onnx_runtime_backend.md), [Torch-TensorRT](../backends/torch_tensorrt_jit_backend.md), [TorchAO](../backends/torchao_backend.md), [Inductor](../backends/torch_inductor_jit_backend.md)
- Explore [Deployment Guide](../deployment/deployment.md)
- Review [AOT Tuning](../aot_tuning.md) for strategy usage
