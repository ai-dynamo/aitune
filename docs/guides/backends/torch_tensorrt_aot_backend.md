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

# Torch-TensorRT AOT Backend Guide

The Torch-TensorRT AOT (Ahead-Of-Time) backend compiles models using `torch_tensorrt.compile()` and saves the compiled model for later use. This approach is ideal for production deployments where compilation happens once during tuning.

## Overview

- **AOT Compilation**: Compiles during tuning, not at runtime
- **Model Persistence**: Compiled model is saved and loaded
- **Fast Startup**: No compilation overhead at inference time
- **Production Ready**: Deterministic performance
- **Multiple IR Support**: dynamo, torchscript, or fx

## Quick Start

```python
from aitune.torch.backend import TorchTensorRTAotBackend, TorchTensorRTAotBackendConfig
from torch_tensorrt.dynamo import CompilationSettings
import aitune.torch as ait

# Configure backend
config = TorchTensorRTAotBackendConfig(
    ir="dynamo",
    compile_config=CompilationSettings(enabled_precisions={torch.float16}),
)
backend = TorchTensorRTAotBackend(config)

# Use in tuning
from aitune.torch.tune_strategy import OneBackendStrategy
strategy = OneBackendStrategy(backend=backend)

model = ait.Module(model, "my-model", strategy=strategy)
ait.tune(model, input_data)

# Save compiled model
ait.save(model, "model.ait")

# Later: Load and use
ait.load(model, "model.ait")
```

## Configuration Options

### TorchTensorRTAotBackendConfig

```python
@dataclass
class TorchTensorRTAotBackendConfig(BackendConfig):
    ir: IRType = "dynamo"
    compile_config: TorchTensorRTConfig
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL
```

### ir

Intermediate representation to use:

```python
# Dynamo (recommended)
config = TorchTensorRTAotBackendConfig(ir="dynamo")

# TorchScript
config = TorchTensorRTAotBackendConfig(ir="ts")

# FX
config = TorchTensorRTAotBackendConfig(ir="fx")
```

**Options**:

- `"dynamo"` (default): Modern, best compatibility
- `"ts"`: TorchScript, legacy models
- `"fx"`: FX graph, experimental

### compile_config

Compilation settings:

```python
from torch_tensorrt.dynamo import CompilationSettings

config = TorchTensorRTAotBackendConfig(
    compile_config=CompilationSettings(
        enabled_precisions={torch.float16},
        workspace_size=1 << 30,
        debug=False,
    )
)
```

### pickle_protocol

Protocol for saving compiled model:

```python
config = TorchTensorRTAotBackendConfig(
    pickle_protocol=4,  # Default
)
```

## Complete Examples

### Example 1: Production Deployment

```python
import torch
import aitune.torch as ait
from aitune.torch.backend import TorchTensorRTAotBackend, TorchTensorRTAotBackendConfig
from torch_tensorrt.dynamo import CompilationSettings

# Load model
model = YourModel()
model.eval().cuda()

# Configure for production
config = TorchTensorRTAotBackendConfig(
    ir="dynamo",
    compile_config=CompilationSettings(
        enabled_precisions={torch.float16},
        workspace_size=2 << 30,  # 2GB
    )
)
backend = TorchTensorRTAotBackend(config)

# Tune and save
wrapped_model = ait.Module(model, "production-model", strategy=OneBackendStrategy(backend))
ait.tune(wrapped_model, calibration_data)
ait.save(wrapped_model, "production_model.ait")

# Deploy: Load on target machine
deployed_model = YourModel()
ait.load(deployed_model, "production_model.ait")
output = deployed_model(input_data)
```


## Saving and Loading

### Save Compiled Model

```python
# After tuning
ait.save(wrapped_model, "model.ait")
```

This creates:
- `model.ait`: Checkpoint file with compiled model
- `model_sha256_sums.txt`: Checksums for verification

### Load Compiled Model

```python
# Create fresh model instance
model = YourModel()
model.eval().cuda()

# Load compiled version
ait.load(model, "model.pt")

# Use immediately
output = model(input_data)
```

## AOT vs JIT Comparison

| Feature | AOT Backend | JIT Backend |
|---------|-------------|-------------|
| **When Compiled** | During `tune()` | At first inference |
| **Saved?** | Yes (.pt file) | No |
| **Load Time** | Fast | Fast + compilation |
| **First Inference** | Fast | Slow (compiling) |
| **Recompilation** | No | On shape changes |
| **Best For** | Production | Experimentation |

### Understanding JIT/AOT Tuning vs. JIT/AOT Backends

It's important to note that AITune has two independent concepts:

1. **Tuning Mode** (JIT/AOT): How you trigger tuning (automatic vs. explicit)
2. **Backend Type** (JIT/AOT): How the model gets compiled (runtime vs. saved)

These can be combined in any way. For a detailed explanation of all four combinations and when to use each, see the [Understanding JIT/AOT Terminology](torch_tensorrt_jit_backend.md#understanding-jitaot-terminology) section in the Torch-TensorRT JIT Backend guide.

**Quick Summary**:

- **AOT Tuning + AOT Backend** ✅ Recommended for production (explicit control + saved model)
- **AOT Tuning + JIT Backend**: Explicit control but recompiles on load
- **JIT Tuning + AOT Backend**: Automatic discovery but no artifact reuse between runs
- **JIT Tuning + JIT Backend**: Automatic discovery with runtime compilation

## Best Practices

1. **Use Dynamo IR**: Most compatible with modern PyTorch
2. **Calibration Data**: Use representative data during tuning
3. **Verify After Load**: Test loaded model before deployment
4. **Version Control**: Track both source code and .ait files
5. **GPU Compatibility**: Compile on the same or a compatible GPU as deployment


## Troubleshooting

### Issue: Load fails on different GPU

**Cause**: Engine compiled for different GPU architecture.

**Solution**: Recompile on target GPU or use hardware compatibility level in TensorRT backend.

### Issue: Large .ait file size

**Cause**: Compiled model includes engine binaries.

**Solution**: This is expected. Use compression if needed:

```bash
gzip model.ait
```

### Issue: Compilation takes too long

**Solution**: Reduce workspace size or use timing cache:

```python
compile_config = CompilationSettings(
    enabled_precisions={torch.float16},
    workspace_size=512 << 20,  # Reduce to 512MB
)
```

## Next Steps

- Learn about [TorchTensorRT JIT Backend](torch_tensorrt_jit_backend.md)
- Explore [Deployment Guide](../deployment/deployment.md)
- Review [TensorRT Backend](tensorrt_backend.md) for more options
