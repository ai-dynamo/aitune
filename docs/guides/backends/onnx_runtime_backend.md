---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "ONNXRuntime Backend Guide"
---

The ONNXRuntime backend exports a wrapped PyTorch module to ONNX, or uses an existing `OnnxModule` graph directly, and runs inference with ONNX Runtime on NVIDIA GPUs. It is useful when you want an ONNX artifact, broad operator coverage through ONNX Runtime, or a TensorRT execution-provider path without using the TensorRT backend directly.

## Quick Start

```python
from aitune.torch.backend import ONNXExecutionProvider, ONNXRuntimeBackend, ONNXRuntimeBackendConfig
import aitune.torch as ait

config = ONNXRuntimeBackendConfig(
    execution_provider=ONNXExecutionProvider.CUDA,
)
backend = ONNXRuntimeBackend(config)

strategy = ait.OneBackendStrategy(backend=backend)
model = ait.Module(model, "my-model", strategy=strategy)
ait.tune(model, input_data)
```

## Execution Providers

AITune supports NVIDIA GPU-backed ONNX Runtime providers:

| Provider | Behavior |
|---|---|
| `ONNXExecutionProvider.CUDA` | Uses `CUDAExecutionProvider` for standard GPU execution |
| `ONNXExecutionProvider.TENSORRT` | Uses `TensorrtExecutionProvider` with `CUDAExecutionProvider` fallback |

When `execution_provider` is `None`, AITune defaults to `CUDA`.

```python
config = ONNXRuntimeBackendConfig(
    execution_provider=ONNXExecutionProvider.TENSORRT,
)
backend = ONNXRuntimeBackend(config)
```

## Configuration

```python
from dataclasses import dataclass

from aitune.torch.backend import BackendConfig, ONNXExecutionProvider

@dataclass
class ONNXRuntimeBackendConfig(BackendConfig):
    use_dynamo: bool = True
    execution_provider: ONNXExecutionProvider | None = None
    opset_version: int | None = None
```

`use_dynamo` and `opset_version` apply only when exporting PyTorch modules. `OnnxModule` uses its existing graph without re-export.

### use_dynamo

When `True`, AITune exports through `torch.onnx.export(dynamo=True)`, which uses the newer torch export path internally. Set it to `False` to use the classic trace-based exporter when that provides better coverage for a specific model.

### execution_provider

Selects the ONNX Runtime execution provider. Use `CUDA` for standard GPU execution and `TENSORRT` when you want ONNX Runtime to build TensorRT engines for supported subgraphs.

### opset_version

Passes an explicit ONNX opset version to `torch.onnx.export`. Leave it as `None` to use the PyTorch default.

## Runtime Behavior

When tuning a PyTorch module, AITune writes the exported model as `model_raw.onnx` in the backend cache directory. If the ONNX exporter emits an external data file, AITune stores and restores it with the checkpoint.

Inference uses ONNX Runtime IOBinding. Inputs are bound from CUDA tensors, outputs are allocated on CUDA, and output tensors are copied back to PyTorch tensors without a CPU round trip.

For PyTorch modules, dynamic batch and spatial dimensions are derived from the recorded graph spec and passed into export. Provide representative input samples that cover the shapes you expect in production.

## Export a Tuned Artifact

After tuning the wrapped module from the quick start, retrieve its `DeploymentArtifact`:

```python
artifact = model.artifact()
artifact.model.export_files("deployment/model.onnx")

print(artifact.input_names)
print(artifact.output_names)
print(artifact.max_batch_size)
print(artifact.model.format)       # onnx
print(artifact.runtime.name)       # onnxruntime
print(artifact.runtime.options)    # {"execution_provider": "cuda"}
```

`DeploymentArtifact` is a portable description of a tuned executable that deployment code can inspect without
loading the model. It contains:

- `model`: a `ModelFiles` record with the format, main file path, additional relative paths, and format-specific metadata.
- `inputs` / `outputs`: ordered bounded tensor specifications.
- `runtime`: a `RuntimeConfig` record with the runtime name and its selected options.

`artifact.model.path` identifies the main ONNX file in the cache. `artifact.model.additional_files` contains relative paths to required files, such as external weights. `artifact.model.files` lists all source paths for storage or upload tools. For the ONNX backend, `artifact.runtime.options["execution_provider"]` is `"cuda"` or `"tensorrt"`; these are portable strings, rather than Torch configuration objects.

`artifact.model.export_files()` copies directly to the destination and creates missing directories. It preserves additional files' relative paths even when the main file is renamed. Existing files are overwritten, and a failed copy can leave an incomplete export. Keep the source files unchanged while copying; the deployment application handles staging, publication, and remote transfers.

The input and output records describe executable tensor names, dtypes, shape bounds, and known batch axes. Batch-size discovery records the expanded input and output batch bounds in `GraphSpec`. The artifact uses the graph's effective output bounds, which also account for explicit input batch ranges. Other output dimensions retain their observed bounds, so representative samples are still needed for varying spatial or sequence dimensions. `max_batch_size` is `None` when the complete interface cannot be described with a shared batch axis supporting batch size one.

`model.artifact()` constructs and validates the deployment record when called. Activation retains the finalized tensor interface, so the record can also be generated after deactivation without reopening a session. AITune creates a separate backend for each graph detected from sample metadata. Because `model.artifact()` returns one deployment artifact, it is available only when the module contains a single compiled graph. It raises `RuntimeError` for multiple graphs, a backend without artifact support, or an ONNX interface that cannot be represented by the recorded metadata. Successful backend inference does not by itself guarantee an artifact is available.

The API is also available after loading a checkpoint:

```python
ait.save(model, "model.ait")
restored = ait.load(model, "model.ait")
restored.artifact().model.export_files("deployment/restored.onnx")
```

## When to Use

Use ONNXRuntime when:

- You want ONNX Runtime as the runtime target.
- You want to compare CUDA and TensorRT execution providers from the same ONNX export path.
- You need an ONNX artifact as part of your deployment or debugging workflow.

Prefer `TensorRTBackend` when you need direct TensorRT engine control, TensorRT-specific configuration, CUDA Graphs, or Model Optimizer quantization workflows.

## Existing ONNX files

`OnnxModule` wraps an existing ONNX file as a torch module. Its forward method uses ONNX Runtime,
accepts tensors positionally in graph input order or by their original names, and returns a dictionary
keyed by the graph output names.

```python
from aitune.torch.module import OnnxModule

source = OnnxModule("model.onnx")
outputs = source(**{"input.1": input_tensor})
```

Only `ONNXRuntimeBackend` and `TensorRTBackend` support tuning `OnnxModule`. Dynamic strategy resolution selects these
backends automatically; pass an explicit compatible backend list to restrict the candidates further. See
[ONNX Model Tuning](../onnx_tuning.md) for a complete example.

`ONNXRuntimeBackend` and `TensorRTBackend`
use the source file directly instead of exporting it again. Recording derives min/max shapes and batch
axes from samples using the same logic as ordinary torch modules. Explicit `dynamic_shapes` retain their
usual precedence. Python call signatures and graph grouping follow the ordinary torch workflow.

Original tensor names are preserved in `TensorSpec.name`, independently of Python access paths.
For example, a named ONNX input `input.1` has the recorded path `("kwargs", "input.1")`; use this path
for `dynamic_shapes` or `TensorRTProfile.add_input_shape()`. Positional inputs use `("args", index)` paths. Calibration, profiles, runtime bindings,
and checkpoints use the preserved tensor name.

## Troubleshooting

### Issue: ONNX export fails

Try `use_dynamo=False`:

```python
config = ONNXRuntimeBackendConfig(use_dynamo=False)
backend = ONNXRuntimeBackend(config)
```

### Issue: Provider is unavailable

Install an ONNX Runtime GPU package matching your CUDA, cuDNN, and TensorRT environment. ONNX Runtime publishes the current GPU install matrix in its [official install guide](https://onnxruntime.ai/docs/install/#install-onnx-runtime-gpu-cuda-or-tensorrt).

## Next Steps

- Compare with [TensorRT Backend](tensorrt_backend.md) for direct TensorRT engine builds
- Compare with [Torch-TensorRT Backend](torch_tensorrt_aot_backend.md) for PyTorch-native TensorRT compilation
- Review [Tune Strategies](../tune_strategies/tune_strategies.md) for fallback and throughput selection
