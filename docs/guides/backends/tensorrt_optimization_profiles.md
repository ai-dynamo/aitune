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

# TensorRT Optimization Profiles

## Introduction

TensorRT optimization profiles optimize the performance of a TensorRT engine. They are used to profile the performance of a model at different input shapes and batch sizes.

By default, a single profile is generated from the graph spec that supports the minimum and maximum shapes of the input tensors.

## Using samples for profile generation

### Set the number of samples to use for profile generation

You can set the number of samples to use for profile generation by setting the `max_num_samples_stored` in the `aitune.torch.config` module. By default, it is set to 1 as samples are stored for each backend, model module, and each batch size.

```python
from aitune.torch.config import config as global_config
global_config.max_num_samples_stored = float("inf")
# or you can set it to a specific number of samples to use for profile generation
global_config.max_num_samples_stored = 100
```

### Use the `ProfileMode.SAMPLES_USED` mode

You can use the `ProfileMode.SAMPLES_USED` mode to auto-generate multiple profiles from shapes of samples used for tuning.

```python
backend = TensorRTBackend(TensorRTBackendConfig(profiles=ProfileMode.SAMPLES_USED))
```

### Use the correct samples and the right batch sizes during tuning

If you use a different batch size than the one used for profile generation, the model will not be able to run.

NOTE: As samples for a single parameter have different shapes, we are wrapping them in a `DynamicShapeDataset` to handle different shapes.

```python
data1 = torch.randn((3, 224, 224), device=device).to(dtype)
data2 = torch.randn((3, 448, 448), device=device).to(dtype)

global_config.max_num_samples_stored = 4 # 2 samples x 2 batch sizes

backend = TensorRTBackend(TensorRTBackendConfig(profiles=ProfileMode.SAMPLES_USED))
module = Module(model, "toy-model", strategy=OneBackendStrategy(backend).enable_find_max_batch_size(False))

tune(module, DynamicShapeDataset([data1, data2]), batch_sizes=[2, 8], device=device)  # will generate 4 profiles

module(data1.repeat(8, 1, 1, 1))
module(data2.repeat(8, 1, 1, 1))
```

See `tests/functional/pytorch/027_aitune_torch_toy_model_tensorrt_backend.py` for a full example.

## Using your own profiles

You can use your own profiles by setting the `profiles` argument in the `TensorRTBackendConfig` class.

```python
backend = TensorRTBackend(TensorRTBackendConfig(profiles=[
    TensorRTProfile().add_input_shape("args_0", (3, 224, 224), (3, 224, 224), (3, 224, 224)),
    TensorRTProfile().add_input_shape("args_0", (3, 448, 448), (3, 448, 448), (3, 448, 448)),
]))
```

### Getting the argument names

The argument names `args_0`, etc. are the names of the input tensors in the model. You can get them from logs during tuning for the default single profile mode.

```text
INFO - 🎯 Tuning module: `toy-model` (all graphs)
INFO - ------------------------------------------------------------
INFO - 🚀 Tuning graph `0` for module `toy-model`:
INFO -   number of parameters: 0
INFO -   number of layers: 0
INFO -   precisions:
INFO -   graph_spec:
INFO -     input_spec:
 Tensors:
╒═══════════╤════════╤═══════════════════════════════╤══════════════════╤══════════════════╤═══════════════╕
│ Locator   │ Name   │ Shape                         │ Min Shape        │ Max Shape        │ Dtype         │
╞═══════════╪════════╪═══════════════════════════════╪══════════════════╪══════════════════╪═══════════════╡
│ [0]       │ args_0 │ ['batch0', 3, 'dim2', 'dim3'] │ [2, 3, 224, 224] │ [8, 3, 448, 448] │ torch.float32 │
╘═══════════╧════════╧═══════════════════════════════╧══════════════════╧══════════════════╧═══════════════╛
```
