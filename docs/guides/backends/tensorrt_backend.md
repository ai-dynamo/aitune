---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "TensorRT Backend Guide"
---

The TensorRT backend provides highly optimized inference using NVIDIA's TensorRT engine. It offers the best performance for production deployments on NVIDIA GPUs and seamlessly integrates [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) for advanced quantization workflows.

## Overview

The TensorRT backend:

- **High Performance**: Maximum inference speed on NVIDIA GPUs
- **Dynamic Shapes**: Supports optimization profiles for variable input sizes
- **Quantization**: INT8, FP8, INT4, FP16/BF16 autocast, and mixed precision support
- **CUDA Graphs**: Cached per static profile by default, with normal execution for dynamic ranges or capture failure
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

## Export a Tuned Artifact

After tuning, retrieve the same `DeploymentArtifact` contract used by the ONNX backend:

```python
artifact = model.artifact()
artifact.model.export_files("deployment/model.plan")

print(artifact.model.format)  # tensorrt_plan
print(artifact.runtime.name)  # tensorrt
profiles = artifact.model.metadata["optimization_profiles"]
print(artifact.model.metadata["optimization_profile_count"])
print(artifact.runtime.options)
```

Each optimization profile is a dictionary keyed by executable input name, in engine input order.
Each input contains `min_shape`, `opt_shape`, and `max_shape` tuples. These exact ranges are
preserved alongside the overall input bounds; multiple profiles can leave gaps within those bounds.
Runtime options preserve `use_cuda_graphs`, `max_cuda_graphs`, and `cuda_graph_cache_policy` as plain values
for deployment adapters to interpret.

The exported file is the TensorRT plan. The profile metadata is embedded in the deployment record;
AITune's profile sidecar remains part of its checkpoint, and is not needed to execute the exported plan.
`model.artifact()` constructs and validates the deployment record when called. The backend retains
the finalized tensor names and profile metadata after deactivation, so generation does not require
reloading the engine. The method also works after restoring and deploying a checkpoint.

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
    use_cuda_graphs: bool = True
    max_cuda_graphs: int = 8
    cuda_graph_cache_policy: Literal["lru", "lfu"] = "lfu"
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

TensorRT backend supports multiple quantization methods through TensorRT Model Optimizer integration. Use `ONNXAutoCastConfig` for FP16/BF16 mixed precision, `ONNXQuantizationConfig` for ONNX INT8/FP8/INT4 quantization, and `TorchQuantizationConfig` for ModelOpt PyTorch quantization presets.

```python
# FP16/BF16 mixed precision autocast
config = TensorRTBackendConfig(
    quantization_config=ONNXAutoCastConfig(precision="fp16"),
)

# or

# ONNX quantization
config = TensorRTBackendConfig(
    quantization_config=ONNXQuantizationConfig(precision="int8", calibration_method="max"),
)

# or

# ModelOpt PyTorch quantization presets
config = TensorRTBackendConfig(
    quantization_config=TorchQuantizationConfig(quantization_config="FP8_DEFAULT_CFG"),
)
```

For current ModelOpt preset names and version-specific support, use the Model Optimizer [documentation](https://github.com/NVIDIA/Model-Optimizer) for the installed version instead of copying preset lists into AITune docs.

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

CUDA graphs are enabled by default for engines with fixed input shapes and for optimization profiles
where `min == opt == max` for every dimension of every input. Profiles containing a dynamic range use
ordinary TensorRT execution, even when a request happens to match their optimum shape. Inputs used as
shape tensors also use ordinary execution because fixed dimensions do not guarantee fixed shape values.

Each eligible profile is captured lazily when admitted to the cache. Its graph, execution context, input buffers,
and output allocator are cached together while sharing the engine. Switching back to a previously used
static profile replays its cached graph without recapture while it remains in the cache. The cache holds
up to `max_cuda_graphs` entries (default: 8). The default aged LFU policy can execute an uncached profile
normally to preserve frequently used graphs and avoid repeated capture. The optional LRU policy always
admits uncached profiles, releasing the least recently used graph when full. All TensorRT profiles remain available.
Deactivation releases the cache and usage history; activation starts with an empty cache.

If capture fails, the backend logs a warning, releases the graph cache after execution finishes, and runs the current
and subsequent calls without CUDA graphs. It does not retry capture on shape changes after a failure.
Normal TensorRT execution errors still propagate. A newly loaded backend attempts capture again.

Set `use_cuda_graphs=False` to disable capture explicitly:

```python
config = TensorRTBackendConfig(
    use_cuda_graphs=False,
)
```

**Benefits**:

- Reduced CPU overhead
- Better performance for small models
- Reuse of cached graphs when switching between static profiles

**Limitations**:

- Requests that capture a graph are slower, including recapture after eviction
- Each cached static profile retains its own context and buffers, increasing device memory use
- Not beneficial for very large models

### max_cuda_graphs

Maximum resident CUDA graphs per backend, as a positive integer (default: `8`). This limits cached graphs,
not the number of TensorRT optimization profiles or their total memory usage. It is a count limit, not a
device-memory budget.

```python
config = TensorRTBackendConfig(max_cuda_graphs=16)
```

With LRU, cycling through nine static profiles in an eight-entry cache causes eviction and recapture on
every request after warmup. Default LFU admission avoids admitting profiles on equal frequency, allowing
uncached requests to execute normally instead of repeatedly replacing graphs.

### cuda_graph_cache_policy

Choose `"lfu"` (default) or `"lru"`. LRU always admits an uncached eligible profile, evicting the least
recently used graph when full. LFU uses recent request frequency to protect hot profiles:

- Count every eligible profile request, including requests executed without a cached graph.
- Halve all counts using integer division every 1,024 eligible requests, before counting that request.
  Aging follows request traffic, not elapsed time, so historical popularity fades as new requests arrive.
- Fill free cache slots immediately. When full, choose the least frequently used resident, breaking
  frequency ties by least recent use.
- Admit an incoming profile only if its count is strictly higher than the victim's. Otherwise, execute
  normally on the base context without capturing or evicting a graph. Rejected requests still contribute
  to future admission. This does not disable CUDA graphs or indicate a capture failure.

```python
config = TensorRTBackendConfig(max_cuda_graphs=8, cuda_graph_cache_policy="lfu")
```

Set `cuda_graph_cache_policy="lru"` to always admit the most recently requested profiles.
Saved configurations with an explicit policy retain that choice; configurations without the field use LFU.

Frequency history is bounded by the engine's eligible profiles, independently of the graph cache size.
LFU can protect recurring hot shapes from occasional requests for other shapes, while LRU responds
immediately to changes in the working set. Neither policy accounts for graph memory size or capture cost.
The CUDA graph CI benchmark compares normal execution, LRU, and LFU using identical request sequences,
reporting latency, capture counts, and requests served by graphs. LFU is the default to avoid the repeated
capture overhead observed when the working set exceeds cache capacity. It is not faster for every workload.

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
- **Opt shape**: Maximum observed across all samples
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

These profiles assume the module defines `forward(input)`, so `input` is the path of its top-level tensor parameter:

```python
from aitune.torch.backend.tensorrt import TensorRTProfile

profiles = [
    # Profile for small inputs
    TensorRTProfile()
        .add_input_shape(
            "input",
            min_shape=(1, 3, 224, 224),
            opt_shape=(4, 3, 224, 224),
            max_shape=(8, 3, 224, 224),
        ),
    # Profile for large inputs
    TensorRTProfile()
        .add_input_shape(
            "input",
            min_shape=(1, 3, 512, 512),
            opt_shape=(4, 3, 512, 512),
            max_shape=(8, 3, 512, 512),
        ),
]

config = TensorRTBackendConfig(profiles=profiles)
```

### Finding Input Paths

Input tensor paths are shown in tuning logs. Top-level paths are forward parameter names; nested paths include all
dictionary keys, sequence indices, or attributes:

```text
INFO - 🚀 Tuning graph `0` for module `my-model`:
INFO -   graph_spec:
INFO -     input_spec:
 Tensors:
╒═══════════════╤═════════════════╤═══════════════════════════════╤══════════════════╤══════════════════╤═══════════════╕
│ Access Path   │ Semantic Path   │ Shape                         │ Min Shape        │ Max Shape        │ Dtype         │
╞═══════════════╪═════════════════╪═══════════════════════════════╪══════════════════╪══════════════════╪═══════════════╡
│ input         │ input           │ ['batch0', 3, 'dim2', 'dim3'] │ [2, 3, 224, 224] │ [8, 3, 448, 448] │ torch.float32 │
╘═══════════════╧═════════════════╧═══════════════════════════════╧══════════════════╧══════════════════╧═══════════════╛
```

Use the string shown in the `Path` column as the profile key. For example, the top-level path `input` uses `"input"`,
while a nested dictionary path uses `inputs["tokens"]`.

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
        .add_input_shape("input", min_shape=(1, 3, 224, 224), opt_shape=(4, 3, 224, 224), max_shape=(16, 3, 224, 224))
]
config = TensorRTBackendConfig(profiles=profiles)
```

### Issue: Slow first inference

**Cause**: This is expected when using CUDA graphs (graph capture overhead).

**Solution**: Warmup with a few inference calls before measuring performance.

### Issue: INT8 accuracy drop

**Solution**: Try different quantization algorithms:

```python
# Try 'entropy' instead of 'max'
quantization_config = ONNXQuantizationConfig(
    precision="int8",
    calibration_method="entropy",
)
```

## Best Practices

1. **Use FP16**: Use `ONNXAutoCastConfig(precision="fp16")` for FP16 mixed precision without a full quantization pass
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
