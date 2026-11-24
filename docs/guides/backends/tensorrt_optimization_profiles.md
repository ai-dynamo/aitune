---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "TensorRT Optimization Profiles"
---

A TensorRT optimization profile defines the minimum and maximum input shapes an engine accepts and the preferred
shape TensorRT optimizes for. Set the profile mode or supply profiles directly through `TensorRTBackendConfig.profiles`.

## Use one profile for a shape range

The default, `TensorRTProfileMode.SINGLE`, creates one profile per graph specification. For inputs without
explicit shape definitions, it uses the recorded minimum and maximum shapes and uses the maximum as the preferred
shape. Explicit dynamic shape definitions supply their declared bounds and preferred sizes.

## Using samples for profile generation

### Set the number of samples to use for profile generation

You can set the number of samples to use for profile generation by setting the `max_num_samples_stored` in the `aitune.torch.config` module. By default, it is set to 1 as samples are stored for each backend, model module, and each batch size.

```python
from aitune.torch.config import config as global_config
global_config.max_num_samples_stored = float("inf")
# or you can set it to a specific number of samples to use for profile generation
global_config.max_num_samples_stored = 100
```

### Use the `TensorRTProfileMode.SAMPLES_USED` mode

`TensorRTProfileMode.SAMPLES_USED` creates an exact profile for each recorded shape. When input shapes vary, it also adds a
wide fallback spanning their range, including the discovered maximum batch size when find-max-batch-size is enabled.

When the module defines explicit dynamic shapes, those definitions take precedence and produce one graph-derived
profile instead of sample profiles.

```python
from aitune.torch.backend import TensorRTBackend, TensorRTBackendConfig
from aitune.torch.backend.tensorrt import TensorRTProfileMode

backend = TensorRTBackend(TensorRTBackendConfig(profiles=TensorRTProfileMode.SAMPLES_USED))
```

### Use the correct samples and the right batch sizes during tuning

The fallback accepts unrecorded shapes within its bounds, while the exact profiles are preferred for recorded shapes.
Inputs outside the fallback bounds cannot run.

NOTE: As samples for a single parameter have different shapes, we are wrapping them in a `DynamicShapeDataset` to handle different shapes.

```python
import aitune.torch as ait
from aitune.torch.dataloader import DynamicShapeDataset

data1 = torch.randn((3, 224, 224), device=device).to(dtype)
data2 = torch.randn((3, 448, 448), device=device).to(dtype)

global_config.max_num_samples_stored = 4 # 2 samples x 2 batch sizes

backend = TensorRTBackend(TensorRTBackendConfig(profiles=TensorRTProfileMode.SAMPLES_USED))
module = ait.Module(model, "toy-model", strategy=ait.OneBackendStrategy(backend).enable_find_max_batch_size(False))

ait.tune(module, DynamicShapeDataset([data1, data2]), batch_sizes=[2, 8], device=device)  # 4 exact profiles + fallback

module(data1.repeat(8, 1, 1, 1))
module(data2.repeat(8, 1, 1, 1))
```

See `tests/functional/pytorch/027_aitune_torch_toy_model_tensorrt_backend.py` for a full example.

## Using your own profiles

Supply a list of profiles when you know the shape ranges in advance. This example assumes a module with
`forward(input)` that serves two image resolutions and batches from 1 to 8:

```python
from aitune.torch.backend import TensorRTBackend, TensorRTBackendConfig
from aitune.torch.backend.tensorrt import TensorRTProfile

profiles = [
    TensorRTProfile().add_input_shape("input", (1, 3, 224, 224), (4, 3, 224, 224), (8, 3, 224, 224)),
    TensorRTProfile().add_input_shape("input", (1, 3, 512, 512), (4, 3, 512, 512), (8, 3, 512, 512)),
]
backend = TensorRTBackend(TensorRTBackendConfig(profiles=profiles))
```

For each dimension, set the minimum at or below the preferred size and the preferred size at or below the maximum.
Pick a preferred shape you expect to serve often. Use multiple profiles for distinct shape ranges and test that
the profiles cover your production inputs.

### Getting the input paths

Profile keys use the semantic path rooted at the real `forward` parameter name. For `forward(input)`,
pass `"input"`. For a tensor accessed as `inputs["tokens"]`, pass the tuple `("inputs", "tokens")`.
Look for the `Semantic Path` column in the tuning logs.

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
╒═══════════════╤═════════════════╤═══════════════════════════════╤══════════════════╤══════════════════╤═══════════════╕
│ Access Path   │ Semantic Path   │ Shape                         │ Min Shape        │ Max Shape        │ Dtype         │
╞═══════════════╪═════════════════╪═══════════════════════════════╪══════════════════╪══════════════════╪═══════════════╡
│ input         │ input           │ ['batch0', 3, 'dim2', 'dim3'] │ [2, 3, 224, 224] │ [8, 3, 448, 448] │ torch.float32 │
╘═══════════════╧═════════════════╧═══════════════════════════════╧══════════════════╧══════════════════╧═══════════════╛
```
