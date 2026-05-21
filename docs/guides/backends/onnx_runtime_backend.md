---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "ONNXRuntime Backend Guide"
---

The ONNXRuntime backend exports a wrapped PyTorch module to ONNX and runs inference with ONNX Runtime on NVIDIA GPUs. It is useful when you want an ONNX artifact, broad operator coverage through ONNX Runtime, or a TensorRT execution-provider path without using the TensorRT backend directly.

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

### use_dynamo

When `True`, AITune exports through `torch.onnx.export(dynamo=True)`, which uses the newer torch export path internally. Set it to `False` to use the classic trace-based exporter when that provides better coverage for a specific model.

### execution_provider

Selects the ONNX Runtime execution provider. Use `CUDA` for standard GPU execution and `TENSORRT` when you want ONNX Runtime to build TensorRT engines for supported subgraphs.

### opset_version

Passes an explicit ONNX opset version to `torch.onnx.export`. Leave it as `None` to use the PyTorch default.

## Runtime Behavior

During tuning, AITune writes the exported model as `model_raw.onnx` in the backend cache directory. If the ONNX exporter emits an external data file, AITune stores and restores it with the checkpoint.

Inference uses ONNX Runtime IOBinding. Inputs are bound from CUDA tensors, outputs are allocated on CUDA, and output tensors are copied back to PyTorch tensors without a CPU round trip.

Dynamic batch and spatial dimensions are derived from the recorded graph spec and passed into export. Provide representative input samples that cover the shapes you expect in production.

## When to Use

Use ONNXRuntime when:

- You want ONNX Runtime as the runtime target.
- You want to compare CUDA and TensorRT execution providers from the same ONNX export path.
- You need an ONNX artifact as part of your deployment or debugging workflow.

Prefer `TensorRTBackend` when you need direct TensorRT engine control, TensorRT-specific configuration, CUDA Graphs, or Model Optimizer quantization workflows.

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
