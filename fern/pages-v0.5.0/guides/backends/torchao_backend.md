---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "TorchAO Backend Guide"
---

The TorchAO backend leverages PyTorch's torchao library for quantization-based model tuning. It provides various quantization schemes for weight-only and dynamic quantization.

## Overview

- **Weight-Only Quantization**: INT8, FP8
- **Dynamic Quantization**: INT8, FP8, MXFP8, and NVFP4 with dynamic activations
- **Easy Configuration**: Predefined quantization types
- **Pure PyTorch**: No external dependencies beyond torchao

## Quick Start

```python
from aitune.torch.backend import TorchAOBackend, TorchAOBackendConfig
import aitune.torch as ait

# Configure with FP8 weight-only quantization
config = TorchAOBackendConfig(quantization="fp8wo")
backend = TorchAOBackend(config)

# Use in tuning
strategy = ait.OneBackendStrategy(backend=backend)

model = ait.Module(model, "my-model", strategy=strategy)
ait.tune(model, input_data)
```

## Quantization Types

### Weight-Only Quantization

```python
# INT8 weight-only
config = TorchAOBackendConfig(quantization="int8wo")

# FP8 weight-only (default)
config = TorchAOBackendConfig(quantization="fp8wo")
```

### Dynamic Quantization

```python
# INT8 dynamic (activations + weights)
config = TorchAOBackendConfig(quantization="int8dq")

# FP8 dynamic (activations + weights)
config = TorchAOBackendConfig(quantization="fp8dq")

# Blackwell/Hopper-dependent dynamic quantization presets
config = TorchAOBackendConfig(quantization="mxfp8dq")
config = TorchAOBackendConfig(quantization="nvfp4dq")
```

## Configuration Options

### TorchAOBackendConfig

```python
@dataclass
class TorchAOBackendConfig(BackendConfig):
    fullgraph: bool = False
    dynamic: bool | None = None
    mode: TorchCompileMode | None = "max-autotune"
    quantization: Literal["int8wo", "int8dq", "fp8wo", "fp8dq", "mxfp8dq", "nvfp4dq"] | None = None
    quantization_config: AOBaseConfig | None = None
    filter_fn: Callable[[nn.Module, str], bool] | None = None
```

### Using Predefined Types

```python
config = TorchAOBackendConfig(
    quantization="int8wo",  # Choose quantization type
)
```

### Custom Configuration

```python
from torchao.quantization import Int8WeightOnlyConfig

custom_config = Int8WeightOnlyConfig()

config = TorchAOBackendConfig(
    quantization_config=custom_config,
)
```

Use either `quantization` or `quantization_config`, not both.

### torch.compile Options

TorchAOBackend quantizes the module and then runs it through `torch.compile`.

```python
config = TorchAOBackendConfig(
    quantization="fp8wo",
    fullgraph=True,
    dynamic=None,  # None lets AITune resolve this from graph metadata
    mode="max-autotune",
)
```

Supported `mode` values follow `torch.compile`: `"default"`, `"reduce-overhead"`, `"max-autotune"`, and `"max-autotune-no-cudagraphs"`.

### Filtering Modules

Use `filter_fn` to restrict quantization to compatible submodules. The predicate receives `(module, fqn)` and should return `True` for modules that TorchAO should quantize.

```python
import torch


def linear_only(module: torch.nn.Module, fqn: str) -> bool:
    return isinstance(module, torch.nn.Linear) and "embed" not in fqn


config = TorchAOBackendConfig(
    quantization="fp8dq",
    filter_fn=linear_only,
)
```

## Quantization Comparison

| Type    | Weights | Activations | Memory Reduction | Speed     | Accuracy  |
|---------|---------|-------------|------------------|-----------|-----------|
| int8wo  | INT8    | FP16/FP32   | ~2x              | High      | Better    |
| int8dq  | INT8    | INT8        | ~2x              | Very High | Good      |
| fp8wo   | FP8     | FP16/FP32   | ~2x              | Very High | Excellent |
| fp8dq   | FP8     | FP8         | ~2x              | Very High | Excellent |
| mxfp8dq | MXFP8   | MXFP8       | ~2x              | Very High | Excellent |
| nvfp4dq | NVFP4   | NVFP4       | ~4x              | Very High | Model-dependent |

`mxfp8dq` and `nvfp4dq` require hardware and torchao support for the corresponding formats. AITune defers that validation until backend build time.

## Best Practices

1. **Start with FP8**: Best accuracy/performance trade-off
2. **Use INT8 for Memory**: When memory is critical
3. **Dynamic Quantization**: Better accuracy, slightly higher overhead
4. **Validate Accuracy**: Always test quantized model accuracy
5. **Calibration Data**: Use representative samples

## Troubleshooting

### Issue: Accuracy loss too high

**Solution**: Try less aggressive quantization:

```python
# Instead of int8wo, try fp8wo
config = TorchAOBackendConfig(quantization="fp8wo")
```

### Issue: Not enough speed improvement

**Solution**: Try dynamic quantization:

```python
config = TorchAOBackendConfig(quantization="fp8dq")
```

## Next Steps

- Learn about [TensorRT Backend](tensorrt_backend.md) for maximum performance
- Compare with [Torch Inductor JIT Backend](torch_inductor_jit_backend.md)
- Review [Deployment Guide](../deployment/deployment.md)
