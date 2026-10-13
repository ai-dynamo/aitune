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
    export_hook: Callable[[ExportedProgram], ExportedProgram] | None = None
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

#### export_hook

Read or edit the FX graph before AOT Inductor compiles it. The backend calls the hook once per build,
after `torch.export` and before `aoti_compile_and_package`. The hook receives the
`torch.export.ExportedProgram`. Its `graph_module` is an ATen-level `torch.fx.GraphModule`, and each
node has `nn_module_stack` metadata that maps it to the source module. The hook can edit the graph in
place. It must return the same or a new `ExportedProgram`; any other value fails the `Torch export`
build step with a `TypeError`. The example below replaces each ReLU with an equivalent
`clamp_min`, so the outputs pass the default tuning validation.

```python
import torch
from aitune.torch.backend import TorchInductorAotBackend, TorchInductorAotBackendConfig


def relu_to_clamp_min(exported: torch.export.ExportedProgram) -> torch.export.ExportedProgram:
    for node in exported.graph_module.graph.nodes:
        if node.op == "call_function" and node.target is torch.ops.aten.relu.default:
            node.target = torch.ops.aten.clamp_min.default
            node.args = (node.args[0], 0.0)
    exported.graph_module.recompile()
    return exported


backend = TorchInductorAotBackend(TorchInductorAotBackendConfig(export_hook=relu_to_clamp_min))
```

<Note>
The hook is build-time only. It is not part of `to_dict()` or `to_json()`, so a config restored with
`from_dict()` has no hook. The config `key()` and `describe()` use the hook's module and qualified name,
so they are the same on all ranks and in all processes; give different hooks different names. Tuning
deep-copies the backend but not the hook, so a bound-method hook can store graphs on its object. The hook
must keep the output values and structure of the program unchanged, because the call contract is captured from the
original module. If the hook inserts custom operators, register them before loading the `.pt2`
package with `ait.load`.
</Note>

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

## Deployment Artifact

Request a deployment record from a tuned module and copy its package:

```python
artifact = model.artifact()
package_path = artifact.model.export_files("deployment/model.pt2")
```

The returned `DeploymentArtifact` contains:

- `model.format="pt2"`, the package path, and `model.metadata["structured_call"]`,
  which indicates whether the call uses nested inputs or container outputs.
- `runtime.name="aotinductor"` with empty runtime options. Inductor configuration
  controls compilation and is already reflected in the package.
- Bounded input and output tensor specs named `INPUT__0`, `OUTPUT__0`, and so on,
  in PyTorch's flattened call order.

The backend creates this record only when `artifact()` is called. It retains the
tensor ordering captured during build, so the record can also be requested after
deactivation or directly from a backend restored from a checkpoint, without loading
the compiled runner. Calls with non-tensor leaves currently raise an error when
requesting the record; this does not prevent tuning or inference.

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
