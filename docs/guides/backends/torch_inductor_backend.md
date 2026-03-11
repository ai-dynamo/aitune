<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
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
import torch

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

| Feature          | Inductor  | TensorRT    | TorchAO   |
|------------------|-----------|-------------|-----------|
| **Dependencies** | None      | TensorRT    | torchao   |
| **Setup**        | Easy      | Moderate    | Easy      |
| **Performance**  | Good      | Excellent   | Good      |
| **Quantization** | Limited   | Advanced    | Extensive |
| **Portability**  | Excellent | NVIDIA only | Good      |

## Next Steps

- Compare with [TensorRT Backend](tensorrt_backend.md) for maximum performance
- Explore [TorchAO Backend](torchao_backend.md) for quantization
- Learn about [Tune Strategies](../tune_strategies/tune_strategies.md)
- Review [Deployment Guide](../deployment/deployment.md)
