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

# TorchAO Backend Guide

The TorchAO backend leverages PyTorch's torchao library for quantization-based model tuning. It provides various quantization schemes for weight-only and dynamic quantization.

## Overview

- **Weight-Only Quantization**: INT4, INT8, FP8, FP6, FP5, FP4
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
# INT4 weight-only
config = TorchAOBackendConfig(quantization="int4wo")

# INT8 weight-only
config = TorchAOBackendConfig(quantization="int8wo")

# FP8 weight-only (default)
config = TorchAOBackendConfig(quantization="fp8wo")

# FP6 E3M2
config = TorchAOBackendConfig(quantization="fp6e3m2")

# FP5 E2M2
config = TorchAOBackendConfig(quantization="fp5e2m2")

# FP4 E2M1
config = TorchAOBackendConfig(quantization="fp4e2m1")
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
    quantization="int4wo",  # Choose quantization type
)
```

### Custom Configuration

```python
from torchao.quantization import Int4WeightOnlyConfig

custom_config = Int4WeightOnlyConfig(
    group_size=64,  # Quantization group size
)

config = TorchAOBackendConfig(
    quantization_config=custom_config,
)
```

## Complete Examples

### Example 1: LLM with INT4

```python
import torch
import aitune.torch as ait
from aitune.torch.backend import TorchAOBackend, TorchAOBackendConfig
from transformers import AutoModelForCausalLM

# Load LLM
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval().cuda()

# Configure INT4 quantization
config = TorchAOBackendConfig(quantization="int4wo")
backend = TorchAOBackend(config)

# Tune
wrapped_model = ait.Module(model, "gpt2", strategy=ait.OneBackendStrategy(backend))
ait.tune(wrapped_model, calibration_data)

# Use quantized model
output = wrapped_model(input_ids)
```

### Example 2: Comparing Quantization Schemes

```python
quantization_types = ["int4wo", "int8wo", "fp8wo"]

for quant_type in quantization_types:
    config = TorchAOBackendConfig(quantization=quant_type)
    backend = TorchAOBackend(config)

    # Tune and evaluate
    model_copy = deepcopy(original_model)
    wrapped = ait.Module(model_copy, f"model_{quant_type}", strategy=ait.OneBackendStrategy(backend))
    ait.tune(wrapped, calibration_data)

    # Evaluate accuracy and speed
    accuracy = evaluate(wrapped, test_data)
    speed = benchmark(wrapped, test_data)
    print(f"{quant_type}: accuracy={accuracy:.3f}, speed={speed:.2f}ms")
```

## Quantization Comparison

| Type | Weights | Activations | Memory Reduction | Speed | Accuracy |
|------|---------|-------------|------------------|-------|----------|
| int4wo | INT4 | FP16/FP32 | ~4x | High | Good |
| int8wo | INT8 | FP16/FP32 | ~2x | High | Better |
| int8dq | INT8 | INT8 | ~2x | Very High | Good |
| fp8wo | FP8 | FP16/FP32 | ~2x | Very High | Excellent |
| fp8dq | FP8 | FP8 | ~2x | Very High | Excellent |
| fp6e3m2 | FP6 | FP16/FP32 | ~2.7x | High | Very Good |
| fp5e2m2 | FP5 | FP16/FP32 | ~3.2x | High | Good |
| fp4e2m1 | FP4 | FP16/FP32 | ~4x | High | Moderate |

## Best Practices

1. **Start with FP8**: Best accuracy/performance trade-off
2. **Use INT4 for Memory**: When memory is critical
3. **Dynamic Quantization**: Better accuracy, slightly higher overhead
4. **Validate Accuracy**: Always test quantized model accuracy
5. **Calibration Data**: Use representative samples

## Troubleshooting

### Issue: Accuracy loss too high

**Solution**: Try less aggressive quantization:

```python
# Instead of int4wo, try int8wo or fp8wo
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
