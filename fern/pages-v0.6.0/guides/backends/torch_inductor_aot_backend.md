---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Torch Inductor AOT Backend Guide"
---

The Torch Inductor AOT backend compiles models Ahead-of-Time using PyTorch's AOT Inductor
(`torch._inductor.aoti_compile_and_package`). The result is a self-contained `.pt2` artifact
that can be saved, loaded, and executed without Python-interpreter overhead.

Requires **PyTorch ≥ 2.6**.

## Overview

- **AOT Compilation**: Model compiled once, loaded as a native artifact at inference time
- **Portable Artifact**: `.pt2` file contains everything needed to run inference
- **Dynamic Shapes**: Batch and spatial dimensions automatically detected and marked dynamic
- **No Python Overhead**: Inference runs through a compiled runner with no Python graph tracing
- **Save / Load**: Artifact persists across sessions via `ait.save` / `ait.load`

## Quick Start

```python
import aitune.torch as ait
from aitune.torch.backend import TorchInductorAotBackend
from aitune.torch.tune_strategy import OneBackendStrategy

backend = TorchInductorAotBackend()
strategy = OneBackendStrategy(backend)

model = ait.Module(model, "my-model", strategy=strategy)
ait.tune(model, input_data, batch_sizes=[1, 2])

# Persist the compiled artifact
ait.save(model, "my_model.ait")
```

Loading in a later session:

```python
import aitune.torch as ait
from aitune.torch.backend import TorchInductorAotBackend
from aitune.torch.tune_strategy import OneBackendStrategy

model = ait.Module(original_model, "my-model", strategy=OneBackendStrategy(TorchInductorAotBackend()))
ait.load(model, "my_model.ait")
```

## Configuration Options

### TorchInductorAotBackendConfig

```python
@dataclass
class TorchInductorAotBackendConfig(BackendConfig):
    inductor_configs: dict[str, Any] | None = None
```

#### inductor_configs

Pass any key from `torch._inductor.config` directly to the compiler:

```python
from aitune.torch.backend import TorchInductorAotBackend, TorchInductorAotBackendConfig

# Default — no extra inductor options
backend = TorchInductorAotBackend()

# Enable max-autotune kernel search
config = TorchInductorAotBackendConfig(inductor_configs={"max_autotune": True})
backend = TorchInductorAotBackend(config=config)

# Coordinate-descent kernel tuning
config = TorchInductorAotBackendConfig(
    inductor_configs={
        "max_autotune": True,
        "coordinate_descent_tuning": True,
    }
)
backend = TorchInductorAotBackend(config=config)
```

See all available keys:

```python
import torch
torch._inductor.list_options()
```

## Dynamic Shapes

By default, dynamic shapes are inferred automatically from the data samples passed to `ait.tune`.

- **Batch axis**: detected when the same tensor dimension varies proportionally with `batch_size`
- **Spatial / sequence axes**: detected when a dimension varies independently of batch size

```python
# Two samples with different spatial sizes → H and W detected as dynamic
data_224 = torch.randn(3, 224, 224, device="cuda")
data_256 = torch.randn(3, 256, 256, device="cuda")

# batch_sizes=[1] keeps samples separate (no cross-shape stacking)
ait.tune(module, [data_224, data_256], batch_sizes=[1])
```

<Note>
When input samples have different spatial sizes, use `batch_sizes=[1]` to prevent the
data loader from stacking tensors of mismatched shapes.

</Note>
The backend uses `torch.export.Dim.AUTO` for spatial / sequence axes, letting PyTorch infer
valid ranges and divisibility constraints from the model automatically.

When recorded samples do not cover the full production range, provide explicit bounded dimensions on `ait.Module`.
See [User-provided dynamic shapes](../aot_tuning.md#user-provided-dynamic-shapes) and the runnable
[ResNet dynamic-shapes example](../../../examples/ResNet/README.md#user-provided-dynamic-shapes).

## Save and Load

```python
# After tuning
ait.save(model, "model.ait")

# In a new process — original weights still needed for Module wrapping,
# but inference runs entirely through the compiled runner
model = ait.Module(original_model, "my-model",
                   strategy=OneBackendStrategy(TorchInductorAotBackend()))
ait.load(model, "model.ait")
result = model(input_tensor)
```

After `ait.tune` completes, the original module is offloaded to CPU to free GPU memory.
The compiled `.pt2` runner is fully self-contained for inference.

## Comparison with Torch Inductor JIT Backend

| Feature                  | AOT Backend                     | JIT Backend                  |
|--------------------------|---------------------------------|------------------------------|
| **Compilation time**     | At `tune()` call                | At first inference call      |
| **Artifact persistence** | Yes (`.pt2` file)               | No                           |
| **Python overhead**      | None at inference               | Minimal (compiled graph)     |
| **Dynamic shapes**       | Inferred or explicitly configured | Configurable via `dynamic=`  |
| **Save / Load**          | Supported                       | Supported (recompiles)       |
| **Requires PyTorch**     | ≥ 2.6                           | Any                          |

## Next Steps

- Compare with [Torch Inductor JIT Backend](torch_inductor_jit_backend.md) for JIT compilation
- Compare with [TensorRT Backend](tensorrt_backend.md) for maximum performance
- Learn about [Tune Strategies](../tune_strategies/tune_strategies.md)
- Review [Deployment Guide](../deployment/deployment.md)
