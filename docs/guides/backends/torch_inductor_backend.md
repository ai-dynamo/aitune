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

# Torch Inductor Backend Guide

The Torch Inductor backend uses PyTorch's built-in compiler (`torch.compile` with `backend="inductor"`) for model tuning. It provides automatic kernel fusion and optimization without external dependencies.

## Overview

- **Pure PyTorch**: No external dependencies
- **Automatic Optimization**: Kernel fusion and code generation
- **Multiple Modes**: Default, reduce-overhead, max-autotune
- **Dynamic Shapes**: Configurable dynamic shape support
- **Cross-Platform**: Works on CPU and CUDA

## Quick Start

```python
from aitune.torch.backend import TorchInductorBackend, TorchInductorBackendConfig
import aitune.torch as ait

# Configure backend
config = TorchInductorBackendConfig(mode="max-autotune")
backend = TorchInductorBackend(config)

# Use in tuning
from aitune.torch.tune_strategy import OneBackendStrategy
strategy = ait.OneBackendStrategy(backend=backend)

model = ait.Module(model, "my-model", strategy=strategy)
ait.tune(model, input_data)
```

## Configuration Options

### TorchInductorBackendConfig

```python
@dataclass
class TorchInductorBackendConfig(BackendConfig):
    fullgraph: bool = False
    dynamic: bool | None = None
    mode: str | None = None
    options: dict | None = None
    autocast_enabled: bool = False
    autocast_dtype: torch.dtype | None = None
```

### mode

Predefined optimization modes:

```python
# Default mode (balanced)
config = TorchInductorBackendConfig(mode="default")

# Reduce Python overhead with CUDA graphs
config = TorchInductorBackendConfig(mode="reduce-overhead")

# Maximum auto-tuning
config = TorchInductorBackendConfig(mode="max-autotune")

# Max autotune without CUDA graphs
config = TorchInductorBackendConfig(mode="max-autotune-no-cudagraphs")
```

**Mode Details**:

- **default**: Good balance, general purpose
- **reduce-overhead**: Uses CUDA graphs for small batches, reduces Python overhead
- **max-autotune**: Leverages Triton for matmul/conv, enables CUDA graphs
- **max-autotune-no-cudagraphs**: Like max-autotune but without CUDA graphs

### fullgraph

Require complete graph capture:

```python
config = TorchInductorBackendConfig(
    fullgraph=True,  # Error if graph breaks occur
    mode="max-autotune",
)
```

### dynamic

Control dynamic shape behavior:

```python
# Always generate dynamic kernels
config = TorchInductorBackendConfig(dynamic=True)

# Never generate dynamic kernels (always specialize)
config = TorchInductorBackendConfig(dynamic=False)

# Auto-detect (default)
config = TorchInductorBackendConfig(dynamic=None)
```

### options

Custom inductor options:

```python
# See all options: torch._inductor.list_options()
config = TorchInductorBackendConfig(
    options={
        "triton.cudagraphs": True,
        "max_autotune": True,
        "coordinate_descent_tuning": True,
    }
)
```

**Note**: Cannot use both `mode` and `options`.

### autocast

Enable automatic mixed precision:

```python
config = TorchInductorBackendConfig(
    mode="max-autotune",
    autocast_enabled=True,
    autocast_dtype=torch.float16,
)
```

## Complete Examples

### Example 1: ResNet with Max Autotune

```python
import torch
import torchvision.models as models
import aitune.torch as ait
from aitune.torch.backend import TorchInductorBackend, TorchInductorBackendConfig

# Load model
model = models.resnet50(pretrained=True)
model.eval().cuda()

# Configure for maximum performance
config = TorchInductorBackendConfig(
    mode="max-autotune",
    autocast_enabled=True,
    autocast_dtype=torch.float16,
)
backend = TorchInductorBackend(config)

# Tune
wrapped_model = ait.Module(model, "resnet50", strategy=OneBackendStrategy(backend))
input_data = torch.randn(8, 3, 224, 224, device="cuda")
ait.tune(wrapped_model, input_data)

# Inference
output = wrapped_model(input_data)
```

### Example 2: Low Latency with CUDA Graphs

```python
config = TorchInductorBackendConfig(
    mode="reduce-overhead",  # Uses CUDA graphs
    dynamic=False,  # Specialize for fixed shapes
)
backend = TorchInductorBackend(config)

# Tune with fixed batch size
wrapped_model = ait.Module(model, "low-latency", strategy=OneBackendStrategy(backend))
input_data = torch.randn(1, 3, 224, 224, device="cuda")
ait.tune(wrapped_model, input_data)
```

### Example 3: Dynamic Shapes

```python
config = TorchInductorBackendConfig(
    mode="default",
    dynamic=True,  # Enable dynamic shapes
)
backend = TorchInductorBackend(config)

# Tune with multiple batch sizes
input_data = [
    torch.randn(1, 3, 224, 224, device="cuda"),
    torch.randn(4, 3, 224, 224, device="cuda"),
    torch.randn(8, 3, 224, 224, device="cuda"),
]
wrapped_model = ait.Module(model, "dynamic", strategy=OneBackendStrategy(backend))
ait.tune(wrapped_model, input_data)

# Works with any batch size
output = wrapped_model(torch.randn(6, 3, 224, 224, device="cuda"))
```

## Mode Comparison

| Mode | CUDA Graphs | Auto-tuning | Overhead | Best For |
|------|-------------|-------------|----------|----------|
| default | No | Moderate | Normal | General use |
| reduce-overhead | Yes | Moderate | Low | Low latency, small batch |
| max-autotune | Yes | Aggressive | Low | Maximum performance |
| max-autotune-no-cudagraphs | No | Aggressive | Normal | Max performance, variable shapes |

## Debugging

### Enable Logging

```python
# Set environment variables before running
import os
os.environ['TORCH_LOGS'] = 'dynamic,perf_hints,graph_breaks'

# Then run tuning
ait.tune(wrapped_model, input_data)
```

### Check Optimizations

```python
# See what mode does
import torch
print(torch._inductor.list_mode_options())

# See all available options
print(torch._inductor.list_options())
```

## Best Practices

1. **Start with max-autotune**: Best performance for most models
2. **Use reduce-overhead**: For latency-critical applications
3. **Enable Autocast**: Free performance boost with FP16
4. **Dynamic Shapes**: Only when necessary (adds overhead)
5. **Warmup**: Run a few iterations before benchmarking

## Troubleshooting

### Issue: Graph breaks

**Check where breaks occur**:

```bash
TORCH_LOGS=graph_breaks python your_script.py
```

**Solution**: Use `fullgraph=False` (default) to allow partial compilation.

### Issue: Slow compilation

**Solution**: Reduce auto-tuning:

```python
config = TorchInductorBackendConfig(mode="default")
```

### Issue: Not using CUDA graphs

**Check logs**:

```bash
TORCH_LOGS=perf_hints python your_script.py
```

**Common causes**: Input mutations, unsupported operations

### Issue: Variable shape recompilations

**Solution**: Enable dynamic shapes:

```python
config = TorchInductorBackendConfig(
    mode="default",
    dynamic=True,
)
```

## Comparison with Other Backends

| Feature | Inductor | TensorRT | TorchAO |
|---------|----------|----------|---------|
| **Dependencies** | None | TensorRT | torchao |
| **Setup** | Easy | Moderate | Easy |
| **Performance** | Good | Excellent | Good |
| **Quantization** | Limited | Advanced | Extensive |
| **Portability** | Excellent | NVIDIA only | Good |

## Next Steps

- Compare with [TensorRT Backend](tensorrt_backend.md) for maximum performance
- Explore [TorchAO Backend](torchao_backend.md) for quantization
- Learn about [Tune Strategies](../tune_strategies/tune_strategies.md)
- Review [Deployment Guide](../deployment/deployment.md)
