---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Torch-TensorRT JIT Backend Guide"
---

The Torch-TensorRT JIT backend integrates TensorRT acceleration through `torch.compile(backend="torch_tensorrt")`. This provides a seamless JIT (Just-In-Time) compilation experience without needing intermediate model formats.

## Overview

- **JIT Compilation**: Compiles at runtime using `torch.compile`
- **No Intermediate Formats**: No ONNX export or separate engine files
- **PyTorch Native**: Stays within the PyTorch ecosystem
- **Dynamic Recompilation**: Automatically recompiles on shape changes
- **FP16 Support**: Built-in mixed precision support

## Quick Start

```python
from aitune.torch.backend import TorchTensorRTJitBackend, TorchTensorRTJitBackendConfig, TorchTensorRTConfig
import aitune.torch as ait
import torch

# Configure backend
config = TorchTensorRTJitBackendConfig(
    compile_config=TorchTensorRTConfig(),
)
backend = TorchTensorRTJitBackend(config)

# Use in tuning
strategy = ait.OneBackendStrategy(backend=backend)

model = ait.Module(model, "my-model", strategy=strategy)
ait.tune(model, input_data)
```

## Configuration Options

### TorchTensorRTJitBackendConfig

```python
@dataclass
class TorchTensorRTJitBackendConfig(BackendConfig):
    compile_config: TorchTensorRTConfig
    fullgraph: bool = False
    dynamic_shapes: bool | None = None
    autocast_enabled: bool = False
    autocast_dtype: torch.dtype | None = None
```

### compile_config

TensorRT compilation settings from torch_tensorrt:

```python
from torch_tensorrt.dynamo import CompilationSettings

config = TorchTensorRTJitBackendConfig(
    compile_config=CompilationSettings(
        workspace_size=1 << 30,  # 1GB
    )
)
```

By default, the engine matches the model's loaded dtype. To request FP16 kernels via the legacy weak-typing path (deprecated in TensorRT 10.12), pair `enabled_precisions` with `use_explicit_typing=False`:

```python
config = TorchTensorRTJitBackendConfig(
    compile_config=CompilationSettings(
        enabled_precisions={torch.float16},
        use_explicit_typing=False,
    )
)
```

**Common options**:

- `workspace_size`: Maximum workspace memory in bytes
- `enabled_precisions`: Kernel dtype precisions TRT may pick (requires `use_explicit_typing=False`; legacy weak typing)
- `use_explicit_typing`: Respect graph dtypes (default `True`); set `False` to enable legacy weak typing

### fullgraph

Require the entire function to be captured in a single graph.

```python
config = TorchTensorRTJitBackendConfig(
    fullgraph=True,  # Raise error if graph breaks occur
)
```

**Use cases**:

- `False` (default): Allow partial compilation
- `True`: Ensure complete compilation or fail

### dynamic_shapes

Enable dynamic shape tracing:

```python
config = TorchTensorRTJitBackendConfig(
    dynamic_shapes=True,  # Enable dynamic shapes
)
```

**Options**:

- `True`: Generate dynamic kernels up-front
- `False`: Always specialize
- `None` (default): Auto-detect and recompile

### autocast_enabled

Enable automatic mixed precision:

```python
config = TorchTensorRTJitBackendConfig(
    autocast_enabled=True,
    autocast_dtype=torch.float16,
)
```

## JIT vs AOT Torch-TensorRT

| Feature           | JIT Backend                   | AOT Backend                   |
|-------------------|-------------------------------|-------------------------------|
| **Compilation**   | Runtime (first inference)     | Ahead-of-time (during tuning) |
| **Model Storage** | Not saved separately          | Saved                         |
| **Startup Time**  | Slower (compilation overhead) | Faster (pre-compiled)         |
| **Flexibility**   | Auto-recompiles on changes    | Fixed after compilation       |
| **Use Case**      | Development, experimentation  | Production deployment         |

## Understanding JIT/AOT Terminology

It's important to distinguish between two uses of "JIT" and "AOT" in AITune:

### AITune Tuning Modes

- **Ahead-of-Time Tuning**: The declarative approach using `inspect()`, `wrap()`, and `tune()`
  - You explicitly select modules to tune
  - Full control over the tuning process
  - Works with any backend (JIT or AOT)

- **Just-in-Time Tuning**: The automatic approach using environment variables or imports
  - No code changes required
  - AITune automatically discovers and tunes modules
  - Works with any backend (JIT or AOT)

### Torch-TensorRT Backend Types

- **TorchTensorRTJitBackend** (this page): Uses `torch.compile(backend="torch_tensorrt")`
  - Compiles at runtime on first inference
  - Does not save compiled artifacts separately
  - Recompiles automatically on shape changes

- **TorchTensorRTAotBackend**: Uses `torch_tensorrt.compile()`
  - Compiles during the `tune()` call
  - Saves compiled model to disk
  - Fixed compilation (no automatic recompilation)

### Combining Them

You can use **any combination**:

```python
# AOT Tuning + JIT Backend
# Explicit tuning with runtime compilation
wrapped_model = ait.Module(model, "model", strategy=ait.OneBackendStrategy(TorchTensorRTJitBackend()))
ait.tune(wrapped_model, data)
# the tuned model can be saved but during loading JIT backend will tune again

# AOT Tuning + AOT Backend
# Explicit tuning with ahead-of-time compilation (saved model)
wrapped_model = ait.Module(model, "model", strategy=ait.OneBackendStrategy(TorchTensorRTAotBackend()))
ait.tune(wrapped_model, data)
# the tuned model can be saved, during loading AOT backend will be load from disk

# JIT Tuning + JIT Backend
# Automatic tuning with runtime compilation
from aitune.torch import jit_config
from aitune.torch.tune_strategy import FirstWinsStrategy
jit_config.strategy = FirstWinsStrategy(backends=[TorchTensorRTJitBackend()])
```

```bash
export AUTOWRAPT_BOOTSTRAP=aitune_enable_jit_tuning
# each time you start the script JIT tuning starts all over again, JIT backend will tune again the module
```

```python
# JIT Tuning + AOT Backend
# Automatic tuning with ahead-of-time compilation
from aitune.torch import jit_config
from aitune.torch.tune_strategy import FirstWinsStrategy
jit_config.strategy = FirstWinsStrategy(backends=[TorchTensorRTAotBackend()])
```

```bash
export AUTOWRAPT_BOOTSTRAP=aitune_enable_jit_tuning
# each time you start the script JIT tuning starts all over again, currently AITune does not reuse past AOT backend artifact, it will start AOT tuning from scratch
```

**Key Takeaway**: AITune's tuning mode (JIT/AOT) is independent from the backend type (JIT/AOT). Choose based on your needs:

- **Tuning mode**: How you want to control tuning (automatic vs explicit)
- **Backend type**: How the model gets compiled and stored (runtime vs saved)

## Best Practices

1. **Use FP16**: Enable FP16 for better performance
2. **Dynamic Shapes**: Enable if input sizes vary frequently
3. **Fullgraph for Production**: Use `fullgraph=True` to catch issues early
4. **Warmup**: Run a few inference calls before benchmarking

## Troubleshooting

### Issue: Compilation fails

**Solution**: Try with partial compilation:

```python
config = TorchTensorRTJitBackendConfig(
    fullgraph=False,  # Allow partial compilation
)
```

### Issue: Slow first inference

**Cause**: JIT compilation happens on the first run.

**Solution**: This is expected. Subsequent inferences will be fast.

## Next Steps

- Compare with [Torch-TensorRT AOT Backend](torch_tensorrt_aot_backend.md)
- Learn about [TensorRT Backend](tensorrt_backend.md)
- Explore [Tune Strategies](../tune_strategies/tune_strategies.md)
