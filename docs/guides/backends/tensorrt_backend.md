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

# TensorRT Backend Guide

The TensorRT backend provides highly optimized inference using NVIDIA's TensorRT engine. It offers the best performance for production deployments on NVIDIA GPUs and seamlessly integrates [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) for advanced quantization workflows.

## Overview

The TensorRT backend:

- **High Performance**: Maximum inference speed on NVIDIA GPUs
- **Dynamic Shapes**: Supports optimization profiles for variable input sizes
- **Quantization**: INT8, FP16, and mixed precision support
- **CUDA Graphs**: Optional CUDA graph capture for reduced CPU overhead
- **Model Optimizer Integration**: Advanced quantization via TensorRT Model Optimizer
- **Flexible Export**: Supports both Dynamo and script-based ONNX export

## Quick Start

### Basic Usage

```python
from aitune.torch.backend import TensorRTBackend, TensorRTBackendConfig, ONNXAutoCastConfig, ONNXQuantizationConfig, TorchQuantizationConfig
import aitune.torch as ait

# Configure TensorRT backend
config = TensorRTBackendConfig(use_dynamo=True)
backend = TensorRTBackend(config)

# Use with tuning
from aitune.torch.tune_strategy import OneBackendStrategy
strategy = OneBackendStrategy(backend=backend)

model = ait.Module(model, "my-model", strategy=strategy)
ait.tune(model, input_data)
```

### With FP16 Precision

```python
config = TensorRTBackendConfig(
    quantization_config=ONNXAutoCastConfig(precision="fp16"),
    workspace_size=1 << 30,  # 1GB workspace
)
backend = TensorRTBackend(config)
```

### With CUDA Graphs

```python
config = TensorRTBackendConfig(
    use_cuda_graphs=True,  # Enable CUDA graphs
)
backend = TensorRTBackend(config)
```

## Configuration Options

### TensorRTBackendConfig

```python
@dataclass
class TensorRTBackendConfig(BackendConfig):
    use_dynamo: bool = True
    workspace_size: int | None = None
    opset_version: int | None = None
    optimization_level: int | None = None
    compatibility_level: int | None = None
    timing_cache: Path | None = None
    profiles: ProfileMode | list[TensorRTProfile] = ProfileMode.SINGLE
    device: str = "cuda"
    quantization_config: ONNXAutoCastConfig | ONNXQuantizationConfig | TorchQuantizationConfig | None = None
    enable_tf32: bool = True
    use_cuda_graphs: bool = False
```

### use_dynamo

Use `torch.dynamo` for ONNX export (recommended).

```python
# Use Dynamo export (recommended)
config = TensorRTBackendConfig(use_dynamo=True)

# Use script-based export (fallback)
config = TensorRTBackendConfig(use_dynamo=False)
```

**When to use**:

- `True` (default): Better compatibility with modern PyTorch models
- `False`: Legacy models or when Dynamo export fails

### workspace_size

Maximum memory workspace for TensorRT engine building.

```python
config = TensorRTBackendConfig(
    workspace_size=1 << 30,  # 1GB
)

# Or larger for complex models
config = TensorRTBackendConfig(
    workspace_size=4 << 30,  # 4GB
)
```

**Guidelines**:

- Default: TensorRT chooses automatically
- Larger workspace → More optimization opportunities → Longer build time
- Recommended: 1-4GB for most models

### opset_version

ONNX opset version for export.

```python
config = TensorRTBackendConfig(
    opset_version=17,  # Use ONNX opset 17
)
```

**Guidelines**:

- Default: Latest stable opset
- Specify only if you need a particular opset for compatibility

### optimization_level

TensorRT builder optimization level (0-5).

```python
config = TensorRTBackendConfig(
    optimization_level=5,  # Maximum optimization
)
```

**Levels**:

- `0`: No optimization
- `3`: Default (balanced)
- `5`: Maximum optimization (longer build time)

### compatibility_level

Hardware compatibility level for the engine.

```python
import tensorrt as trt

config = TensorRTBackendConfig(
    compatibility_level=trt.HardwareCompatibilityLevel.AMPERE_PLUS,
)
```

**Options**:

- `None`: Optimized for current GPU
- Specific level: Portable across compatible GPUs

### timing_cache

Path to timing cache for faster subsequent builds.

```python
from pathlib import Path

config = TensorRTBackendConfig(
    timing_cache=Path("/path/to/timing_cache.bin"),
)
```

**Benefits**:

- Faster engine rebuilds
- Reuse timing information across builds
- Especially useful during development

### profiles

Optimization profiles for dynamic shapes.

```python
from aitune.torch.backend.tensorrt import ProfileMode, TensorRTProfile

# Single profile (default)
config = TensorRTBackendConfig(
    profiles=ProfileMode.SINGLE,
)

# Multiple profiles from samples
config = TensorRTBackendConfig(
    profiles=ProfileMode.SAMPLES_USED,
)

# Custom profiles
config = TensorRTBackendConfig(
    profiles=[
        TensorRTProfile()
            .add_input_shape("input", (1, 3, 224, 224), (4, 3, 224, 224), (8, 3, 224, 224)),
    ]
)
```

See [Optimization Profiles](#optimization-profiles) section for details.

### device

Device for TensorRT engine.

```python
config = TensorRTBackendConfig(
    device="cuda",  # Default
)
```

### quantization_config

TensorRT backend supports multiple quantization methods through TensorRT Model Optimizer integration.

```python
config = TensorRTBackendConfig(
    quantization_config=ONNXAutoCastConfig(precision="fp16"),
)

# or

config = TensorRTBackendConfig(
    quantization_config=ONNXQuantizationConfig(precision="fp16"),
)

# or

config = TensorRTBackendConfig(
    quantization_config=TorchQuantizationConfig(quantization_config="FP8_DEFAULT_CFG"),
)
```

For a detailed information take a look at Model Optimizer [documentation](https://github.com/NVIDIA/Model-Optimizer).

### enable_tf32

Enable TF32 tensor cores on Ampere+ GPUs.

```python
config = TensorRTBackendConfig(
    enable_tf32=True,  # Default
)
```

**Benefits**:

- Faster FP32 operations on Ampere and newer GPUs
- No accuracy loss for most models
- Recommended to keep enabled

### use_cuda_graphs

Enable CUDA graph capture for inference.

```python
config = TensorRTBackendConfig(
    use_cuda_graphs=True,
)
```

**Benefits**:

- Reduced CPU overhead
- Better performance for small models
- Automatic re-capture on shape changes

**Limitations**:

- First inference is slower (graph capture)
- Shape changes trigger re-capture
- Not beneficial for very large models

## Optimization Profiles

Optimization profiles define the range of input shapes TensorRT will optimize for. They are essential for models with dynamic input sizes.

### Profile Modes

#### SINGLE (Default)

Automatically generates a single profile from recorded samples:

```python
config = TensorRTBackendConfig(
    profiles=ProfileMode.SINGLE,
)
```

- **Min shape**: Minimum observed across all samples
- **Opt shape**: Most common shape
- **Max shape**: Maximum observed across all samples

#### SAMPLES_USED

Generates one profile per unique input shape:

```python
config = TensorRTBackendConfig(
    profiles=ProfileMode.SAMPLES_USED,
)
```

**Important**: Increase `max_num_samples_stored`:

```python
from aitune.torch.config import config as global_config

global_config.max_num_samples_stored = 100  # Or float("inf")
```

**Use case**: When you have distinct input shape categories that need separate optimization.

### Custom Profiles

Define exact optimization profiles:

```python
from aitune.torch.backend.tensorrt import TensorRTProfile

profiles = [
    # Profile for small inputs
    TensorRTProfile()
        .add_input_shape(
            "args_0",
            min_shape=(1, 3, 224, 224),
            opt_shape=(4, 3, 224, 224),
            max_shape=(8, 3, 224, 224),
        ),
    # Profile for large inputs
    TensorRTProfile()
        .add_input_shape(
            "args_0",
            min_shape=(1, 3, 512, 512),
            opt_shape=(4, 3, 512, 512),
            max_shape=(8, 3, 512, 512),
        ),
]

config = TensorRTBackendConfig(profiles=profiles)
```

### Finding Input Names

Input tensor names are shown in tuning logs:

```
INFO - 🚀 Tuning graph `0` for module `my-model`:
INFO -   graph_spec:
INFO -     input_spec:
 Tensors:
╒═══════════╤════════╤═══════════════════════════════╤══════════════════╤══════════════════╤═══════════════╕
│ Locator   │ Name   │ Shape                         │ Min Shape        │ Max Shape        │ Dtype         │
╞═══════════╪════════╪═══════════════════════════════╪══════════════════╪══════════════════╪═══════════════╡
│ [0]       │ args_0 │ ['batch0', 3, 'dim2', 'dim3'] │ [2, 3, 224, 224] │ [8, 3, 448, 448] │ torch.float32 │
╘═══════════╧════════╧═══════════════════════════════╧══════════════════╧══════════════════╧═══════════════╛
```

Use the `Name` column value (e.g., `args_0`) in your profiles.

### Best Practices for Profiles

1. **Min < Opt < Max**: Ensure min ≤ opt ≤ max for all dimensions
2. **Opt = Typical**: Set opt to your most common input size
3. **Range Coverage**: Ensure your runtime inputs fall within [min, max]
4. **Multiple Profiles**: Use for distinct size categories, not slight variations
5. **Test Runtime Shapes**: Verify your production shapes are covered


## Troubleshooting

### Issue: ONNX export fails

**Solution**: Try disabling Dynamo export:

```python
config = TensorRTBackendConfig(use_dynamo=False)
```

### Issue: Engine build fails due to memory

**Solution**: Reduce workspace size:

```python
config = TensorRTBackendConfig(workspace_size=512 << 20)  # 512MB
```

### Issue: Runtime shape not supported

**Error**: `Input shape X exceeds max profile shape Y`

**Solution**: Update profiles to cover your runtime shapes:

```python
profiles = [
    TensorRTProfile()
        .add_input_shape("args_0", min_shape=(1, 3, 224, 224), opt_shape=(4, 3, 224, 224), max_shape=(16, 3, 224, 224))
]
config = TensorRTBackendConfig(profiles=profiles)
```

### Issue: Slow first inference

**Cause**: This is expected when using CUDA graphs (graph capture overhead).

**Solution**: Warmup with a few inference calls before measuring performance.

### Issue: INT8 accuracy drop

**Solution**: Try different quantization algorithms:

```python
# Try 'minmax' or 'entropy' instead of 'max'
quantization_config = QuantizationConfig(
    algorithm="entropy",
    quant_format="int8",
)
```

## Best Practices

1. **Use FP16**: Enable FP16 precision for best performance without accuracy loss
2. **Enable TF32**: Keep `enable_tf32=True` on Ampere+ GPUs
3. **Profile Carefully**: Ensure optimization profiles cover all runtime shapes
4. **Timing Cache**: Use timing cache during development for faster iteration
5. **CUDA Graphs**: Enable for latency-sensitive small models
6. **Workspace Size**: Start with 1-2 GB and, increase if the build fails
7. **Quantization**: Validate accuracy with a representative test set

## Next Steps

- Learn about [Torch-TensorRT JIT Backend](torch_tensorrt_jit_backend.md)
- Learn about [Torch-TensorRT AOT Backend](torch_tensorrt_aot_backend.md)
- Explore [Tune Strategies](../tune_strategies/tune_strategies.md)
- Review [Deployment Guide](../deployment/deployment.md)
