---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "ONNX Model Tuning"
---

`OnnxModule` wraps an existing ONNX file as a PyTorch module so AITune can record inputs, tune it, and run inference.
Only **`ONNXRuntimeBackend`** and **`TensorRTBackend`** support tuning `OnnxModule`. Both use the source ONNX file directly, without exporting it through PyTorch again.

## Load and run an ONNX model

```python
import torch
from aitune.torch.module import OnnxModule

source = OnnxModule("model.onnx")
input_tensor = torch.randn(1, 3, 224, 224, device="cuda")
outputs = source(**{"input.1": input_tensor})
```

This example assumes the graph has a float32 input named `input.1` with shape `[batch, 3, 224, 224]`.
Use your graph's input names, dtypes, and shapes. Keep any external ONNX weight files alongside the model at their referenced paths.

Inputs can be positional in graph input order (`source(input_tensor)`) or passed by their original ONNX names.
Names containing dots or other punctuation work through dictionary unpacking, as above.
Outputs are a dictionary keyed by the original graph output names.

The untuned module runs through ONNX Runtime on CPU or CUDA, based on its input tensors. All inputs must share a device.
The runtime session is created on the first call. To switch an existing session's device, call `source.offload("cpu")` or `source.offload("cuda:0")` before passing inputs on that device.

## Tune with the two supported backends

Use an explicit backend list: the default strategy candidates are not filtered for `OnnxModule` and include incompatible backends.
Tuning with these backends requires an NVIDIA GPU and the corresponding runtime dependencies; see [installation](../learn/install.md).

The following example assumes `model.onnx` accepts dynamic batches of float32 images under the name `input.1`.
Replace the random images with representative preprocessed data for your model.

```python
import torch
import aitune.torch as ait
from aitune.torch.backend import ONNXRuntimeBackend, TensorRTBackend
from aitune.torch.module import OnnxModule

source = OnnxModule("model.onnx")
strategy = ait.MaxThroughputStrategy(
    backends=[ONNXRuntimeBackend(), TensorRTBackend()],
)
strategy.enable_find_max_batch_size(False)
model = ait.Module(source, "onnx-model", strategy=strategy)

# Each dataset item is one image; AITune adds the batch dimension.
dataset = [{"input.1": torch.randn(3, 224, 224)} for _ in range(8)]
ait.tune(model, dataset, batch_sizes=[1, 2, 4, 8], device="cuda")

with torch.inference_mode():
    outputs = model(**{"input.1": torch.randn(1, 3, 224, 224, device="cuda")})
```

This disables maximum-batch-size discovery to keep profiling within the supplied batch sizes.
`MaxThroughputStrategy` compares compatible candidates and selects by throughput. By default, performance validation
can retain the untuned `OnnxModule` baseline if neither candidate improves on it. That baseline runs ONNX Runtime.

To use a single tuning target, replace the strategy with `ait.OneBackendStrategy(ONNXRuntimeBackend())`
or `ait.OneBackendStrategy(TensorRTBackend())`.

## Input shapes and names

Samples determine recorded shape bounds and batch axes, as for ordinary PyTorch modules. Provide at least two different
batch sizes to identify the batch axis, and ensure every sample shape is accepted by the source ONNX graph.
Recording does not turn a fixed-shape ONNX graph into a dynamic one. For a fixed batch dimension, tune only that batch size.

Explicit `dynamic_shapes` take precedence over inferred shapes. Shape configuration uses Python access paths:

- Named input `input.1`: `("kwargs", "input.1")`.
- First positional input: `("args", 0)`.

Use these paths for `dynamic_shapes` or `TensorRTProfile.add_input_shape()`. Original ONNX tensor names are preserved
separately for runtime bindings, calibration, optimization profiles, and checkpoints.
See [TensorRT optimization profiles](backends/tensorrt_optimization_profiles.md) for profile configuration.

## Backend options

| Backend | Execution |
|---|---|
| `ONNXRuntimeBackend` | ONNX Runtime with the CUDA or TensorRT execution provider |
| `TensorRTBackend` | Builds and runs a TensorRT engine directly |

Backends declare supported source formats through `_supported_modules`, a frozenset of `ModuleFormat` values.
The default is `ModuleFormat.TORCH`; ONNX Runtime and TensorRT also support `ModuleFormat.ONNX`.
Unsupported formats are rejected before backend compilation.

For existing ONNX input, export options such as `use_dynamo` and `opset_version` do not re-export or modify the source graph.

`TensorRTBackend` supports `ONNXAutoCastConfig` for FP16/BF16 conversion and `ONNXQuantizationConfig` for ONNX quantization.
`TorchQuantizationConfig` is unsupported for `OnnxModule` and raises an error. Use representative calibration data for quantization.

See the [ONNX Runtime guide](backends/onnx_runtime_backend.md) for execution providers and the
[TensorRT guide](backends/tensorrt_backend.md#quantization_config) for quantization options.
