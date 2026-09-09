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
    use_legacy_exporter: bool | None = None
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

### Serialization exporter

For stateless models, AITune first saves with `retrace=False`, then retries with
`retrace=True` on export specification errors or the known integer-parameter gradient error.
Unrelated failures are propagated without retrying.

Models with input/buffer mutations or output/input aliases use `retrace=True` directly.
An explicit `use_legacy_exporter` setting also uses `retrace=True` directly, passing the
requested exporter when the installed save API supports it. All paths save
`output_format="exported_program"`. Re-export receives the same `torch_tensorrt.Input`
descriptors used for compilation, including min/opt/max shapes, together with explicit
dynamic-shape constraints. Torch-TensorRT prepares the example tensors for saving.
Here, `retrace=True` selects the Torch Export save path; it does not use `torch.jit.trace`
or produce TorchScript. AITune reloads the persisted artifact before accepting the backend,
including stateful validation when configured.
Validation preserves registered model buffers so temporary checks do not advance the state
stored in the artifact or leave the loaded model changed.

By default, `use_legacy_exporter=None` leaves exporter selection to the installed
Torch-TensorRT version. On versions that expose the public `save(use_legacy_exporter=...)`
parameter, request the standard Torch Export implementation explicitly:

```python
config = TorchTensorRTAotBackendConfig(use_legacy_exporter=False)
```

If the installed save API lacks this parameter, AITune omits it and uses the installed
version's default exporter.
Older releases may choose a legacy exporter for dynamic graphs even with `retrace=True`;
use an upstream version supporting the explicit override to control that choice.

AITune does not repair Torch-TensorRT graph metadata or rewrite mutation operators during
serialization. Upstream export failures fail the backend candidate; they do not trigger a
TorchScript or JIT fallback. Changing the exporter can address serialization failures, but
does not establish operator coverage, correctness, or a performance improvement.

See the [Torch-TensorRT save API](https://docs.pytorch.org/TensorRT/py_api/torch_tensorrt.html#torch_tensorrt.save).

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
