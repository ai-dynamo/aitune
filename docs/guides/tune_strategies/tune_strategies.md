<!--
Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

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

# Tune Strategies Guide

Tune strategies determine how AITune selects and configures backends during the tuning process. They provide flexibility in balancing performance, reliability, and tuning time.

## Overview

AITune provides three built-in strategies:

- **OneBackendStrategy**: Uses a single specified backend
- **FirstWinsStrategy**: Tries backends in order, uses the first that succeeds
- **HighestThroughputStrategy**: Profiles all backends, selects the fastest

## Why Backends Can Fail

Not every backend can successfully tune every model. Each backend relies on a different compilation or export technology, and each has its own limitations:

- **TensorRT** requires exporting the model to ONNX. Models with unsupported operators, complex dynamic control flow, or symbolic shape constraints may fail during ONNX export or TensorRT engine building. Memory constraints can also prevent the engine from being built.
- **Torch Inductor** uses `torch.compile`, which may encounter *graph breaks* on unsupported Python constructs or operations, causing partial or failed compilation.
- **TorchAO** applies quantization transformations that may not support all layer types or model architectures.
- **Torch-TensorRT** combines PyTorch's compiler with TensorRT, inheriting potential limitations from both.

Because of these differences, a backend that fails on one model may succeed on another, and vice versa. This is the core motivation behind strategies like `FirstWinsStrategy`: by trying multiple backends in priority order, you get automatic fallback when your preferred backend cannot handle a particular model.

## Choosing a Strategy

Use the table below as a quick decision guide. If you already know a backend is compatible and stable in production, start with `OneBackendStrategy`. If you want a safer default with minimal tuning time, `FirstWinsStrategy` balances reliability and speed. When absolute throughput matters and you can afford longer tuning, choose `HighestThroughputStrategy`.

| Strategy                  | When to Use                     | Tuning Time | Reliability             | Performance        |
|---------------------------|---------------------------------|-------------|-------------------------|--------------------|
| OneBackendStrategy        | Known backend works, production | Fast        | High (if backend works) | Depends on backend |
| FirstWinsStrategy         | Want reliability, quick tuning  | Fast-Medium | Very High               | Good               |
| HighestThroughputStrategy | Maximum performance, have time  | Slow        | High                    | Best               |

## OneBackendStrategy

Uses exactly one backend, failing immediately with the original error if it cannot build. Use this when you have already validated that a backend works and want deterministic, reproducible behavior in production.

!!! note
    `OneBackendStrategy` may look equivalent to `FirstWinsStrategy` with a single backend, but the key difference is error handling: `OneBackendStrategy` raises the backend's original exception on failure, while `FirstWinsStrategy` catches errors and tries the next candidate.

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

Tries backends in priority order and returns the first one that successfully builds and validates. If a backend fails, the strategy moves on to the next candidate instead of aborting. List backends from fastest to most compatible — for example, TensorRT first, then Torch Inductor, and finally `TorchEagerBackend` as a universal fallback that always succeeds (it runs the original, unoptimized model).

### Usage

```python
from aitune.torch.backend import (
    TensorRTBackend,
    TensorRTBackendConfig,
    TorchInductorBackend,
    TorchEagerBackend,
)
import aitune.torch as ait

# List backends in priority order (fastest → most compatible)
backends = [
    TensorRTBackend(config=TensorRTBackendConfig()),  # Best performance, but may not support all models
    TorchInductorBackend(),                            # Good performance, broader compatibility
    TorchEagerBackend(),                               # Always works (no optimization, baseline performance)
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
3. If fails (e.g., unsupported op, export error, memory limit) → tries next backend
4. Repeats until a backend succeeds or all fail

### When to Use

✅ **Good for**:

- Experimentation with unknown or diverse models
- CI/CD pipelines where different models may need different backends
- Maximum reliability (with a universal fallback like `TorchEagerBackend`)
- Quick validation that *something* works before investing in backend-specific tuning

❌ **Not ideal for**:

- Maximum performance (stops at first success, not the fastest)
- When you already know which backend works (use `OneBackendStrategy` instead)
- Detailed performance comparison (use `HighestThroughputStrategy` instead)

### Best Practices

1. **Order by Performance**: Put fastest backends first
2. **Always Include Fallback**: End with `TorchEagerBackend()` for reliability
3. **Similar Configurations**: Use compatible configs (e.g., all FP16)

## HighestThroughputStrategy

Tries all backends, profiles their performance, and selects the fastest.

### Usage

```python
from aitune.torch.backend import (
    TensorRTBackend,
    TensorRTBackendConfig,
    TorchInductorBackendConfig,
    TorchInductorBackend,
    TorchAOBackend,
    TorchAOBackendConfig
)
import aitune.torch as ait

# List all candidate backends
backends = [
    TensorRTBackend(config=TensorRTBackendConfig()),
    TorchInductorBackend(config=TorchInductorBackendConfig(mode="max-autotune")),
    TorchAOBackend(config=TorchAOBackendConfig(quantization="fp8wo")),
]

# Create strategy
strategy = ait.HighestThroughputStrategy(
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
3. Selects backend with highest throughput
4. Returns best performing backend

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

## Best Practices

1. **Development**: Use `FirstWinsStrategy` with fallback
2. **Production**: Use `OneBackendStrategy` with validated backend
3. **Benchmarking**: Use `HighestThroughputStrategy` to find the best option
4. **Always Validate**: Test tuned models before deployment
5. **Cache Results**: Save tuned models to avoid re-tuning

## Next Steps

- Learn about specific backends: [TensorRT](../backends/tensorrt_backend.md), [Torch-TensorRT](../backends/torch_tensorrt_jit_backend.md), [TorchAO](../backends/torchao_backend.md), [Inductor](../backends/torch_inductor_backend.md)
- Explore [Deployment Guide](../deployment/deployment.md)
- Review [AOT Tuning](../aot_tuning.md) for strategy usage
