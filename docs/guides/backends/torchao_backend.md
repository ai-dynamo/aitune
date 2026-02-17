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

# TorchAO Backend Guide

The TorchAO backend leverages PyTorch's torchao library for quantization-based model tuning. It provides various quantization schemes for weight-only and dynamic quantization.

## Overview

- **Weight-Only Quantization**: INT8, FP8
- **Dynamic Quantization**: INT8 and FP8 with dynamic activations
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
```

## Configuration Options

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

## Quantization Comparison

| Type    | Weights | Activations | Memory Reduction | Speed     | Accuracy  |
|---------|---------|-------------|------------------|-----------|-----------|
| int8wo  | INT8    | FP16/FP32   | ~2x              | High      | Better    |
| int8dq  | INT8    | INT8        | ~2x              | Very High | Good      |
| fp8wo   | FP8     | FP16/FP32   | ~2x              | Very High | Excellent |
| fp8dq   | FP8     | FP8         | ~2x              | Very High | Excellent |

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
- Compare with [Torch Inductor Backend](torch_inductor_backend.md)
- Review [Deployment Guide](../deployment/deployment.md)
