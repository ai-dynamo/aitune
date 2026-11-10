---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Torch-TensorRT AOT Backend Guide"
---

The Torch-TensorRT AOT (Ahead-Of-Time) backend exports models with `torch.export.export()`, compiles the exported program with `torch_tensorrt.dynamo.compile()`, and saves the compiled model for later use. This approach is ideal for production deployments where compilation happens once during tuning.

## Overview

- **AOT Compilation**: Compiles during tuning, not at runtime
- **Model Persistence**: Compiled model is saved and loaded
- **Fast Startup**: No compilation overhead at inference time
- **Production Ready**: Deterministic performance
- **Dynamo Export Path**: Uses `torch.export` and the Torch-TensorRT Dynamo frontend

## Quick Start

```python
from aitune.torch.backend import TorchTensorRTAotBackend, TorchTensorRTAotBackendConfig
from torch_tensorrt.dynamo import CompilationSettings
import aitune.torch as ait

# Configure backend
config = TorchTensorRTAotBackendConfig(
    compile_config=CompilationSettings(),
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
    compile_config: TorchTensorRTConfig
    pickle_protocol: int = 5
```

### compile_config

Compilation settings:

```python
from torch_tensorrt.dynamo import CompilationSettings

config = TorchTensorRTAotBackendConfig(
    compile_config=CompilationSettings(
        workspace_size=1 << 30,
    )
)
```

By default, the engine matches the model's loaded dtype. To request FP16 kernels via the legacy weak-typing path (deprecated in TensorRT 10.12), pair `enabled_precisions` with `use_explicit_typing=False`:

```python
config = TorchTensorRTAotBackendConfig(
    compile_config=CompilationSettings(
        enabled_precisions={torch.float16},
        use_explicit_typing=False,
    )
)
```

### pickle_protocol

Protocol for saving compiled model:

```python
config = TorchTensorRTAotBackendConfig(
    pickle_protocol=5,  # Default
)
```

### Export profiles and serialization

Before compilation, AITune checks that the requested TensorRT optimization profiles fit
within the Torch Export shape constraints. Incompatible profiles fail the candidate;
AITune does not silently narrow the requested ranges.

Serialization first exports the compiled graph directly. It preserves scalar and `None`
input metadata so those constants remain serializable. Export specification errors and
the known integer-parameter gradient error trigger a non-strict Torch Export retry;
unrelated errors propagate. That retry uses the captured flat example tensors and
positional dynamic-shape constraints, rather than TensorRT profile descriptors.
AITune reloads the saved artifact before accepting the backend.

## AOT vs JIT Comparison

For a detailed explanation of JIT vs AOT backends, see the [JIT vs AOT Torch-TensorRT](torch_tensorrt_jit_backend.md#jit-vs-aot-torch-tensorrt) section.

## Best Practices

1. **Use Representative Data**: Tune with inputs that match production shapes and dtypes
2. **Verify After Load**: Test loaded model before deployment
3. **Version Control**: Track both source code and .ait files
4. **GPU Compatibility**: Compile on the same or a compatible GPU as deployment

## Troubleshooting

### Issue: Load fails on different GPU

**Cause**: Engine compiled for different GPU architecture.

**Solution**: Recompile on target GPU or use hardware compatibility level in TensorRT backend.

## Next Steps

- Learn about [TorchTensorRT JIT Backend](torch_tensorrt_jit_backend.md)
- Explore [Deployment Guide](../deployment/deployment.md)
- Review [TensorRT Backend](tensorrt_backend.md) for more options
