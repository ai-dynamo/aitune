---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Core Functionalities"
---

## Inspect for AOT tuning

The `inspect` function allows you to analyze PyTorch models and pipelines to understand their structure, parameters, and execution flow. It provides detailed insights into model architecture and helps identify tuning opportunities.

```python
import aitune.torch as ait
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(100, 10)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()

# Inspect the model
ait.inspect(model, dataset)
```

## Inspect for JIT tuning

JIT tuning also has a corresponding `inspect` mode which gathers information about the model/pipeline and allows checking model input and output arguments, hierarchy of the model, etc.

Here is a short snippet how to use it:

```python
# required imports
import aitune.torch.jit.enable_inspection as inspection

# your code goes here
# ...

# you can export report to html file
inspection.save_report("filename.html", "YOUR_MODEL_NAME")
```

## Tune

The `tune` function is the core functionality that automatically tunes your PyTorch models and pipelines for optimal inference performance. It supports various backends and automatically selects the best performing configuration.

```python
import aitune.torch as ait
import torch

# Define your model
model = SimpleModel()

# Wrap the model
model = ait.Module(model)

# Define inference function
def inference_fn(x):
    return model(x)

# Tune the model
ait.tune(
    func=inference_fn,
    dataset=torch.randn(1, 100),
)
```

## Save

The `save` function allows you to persist tuned models for later use. It stores tuned and original module weights together in a single file with a `.ait` extension. Apart from the checkpoint file, there is also a SHA hash file.

```python
# Save the tuned model
import aitune.torch as ait
ait.save(model, "tuned_model.ait")
```

Example output:

```bash
checkpoints/
├── tuned_model
├── tuned_model.ait
└── tuned_model_sha256_sums.txt
```

You can copy the checkpoint file `tuned_model.ait` and SHA sums file to a target host or folder to use it for inference.

*Note:* We recommend deploying the `*.ait` package on the same hardware used for tuning to ensure functional and performance compatibility.

## Load

The `load` function enables you to load previously tuned models from a checkpoint file.

```python
# Load the tuned model
import aitune.torch as ait
tuned_model = ait.load(model, "tuned_model.ait")
```

On first load, the checkpoint file is decompressed and the tuned and original module weights are loaded. Subsequent loads will use the decompressed weights from the same folder.

# Tune Strategies

NVIDIA AITune provides different strategies for selecting the optimal backend configuration. The strategies align with a common interface for the tuning process.

Not every backend can tune every model — each relies on different compilation technology with its own limitations (e.g., ONNX export for TensorRT, graph breaks in Torch Inductor, unsupported layers in TorchAO). Strategies control how AITune handles this.

## FirstWinsStrategy

Tries backends in priority order and returns the first one that succeeds. If a backend fails, the strategy moves on to the next candidate instead of aborting.

```python
from aitune.torch.tune_strategy import FirstWinsStrategy

strategy = FirstWinsStrategy(backends=[TensorRTBackend(), TorchInductorJitBackend()])
```

## OneBackendStrategy

Uses exactly one backend, failing immediately with the original error if it cannot build. Use this when you have already validated that a backend works and want deterministic behavior. Unlike `FirstWinsStrategy` with a single backend, `OneBackendStrategy` surfaces the original exception rather than catching it.

```python
from aitune.torch.tune_strategy import OneBackendStrategy

strategy = OneBackendStrategy(backend=TensorRTBackend())
```

## MaxThroughputStrategy

Profiles all compatible backends and selects the fastest. Use this when maximum throughput matters and you can afford longer tuning time.

```python
from aitune.torch.tune_strategy import MaxThroughputStrategy

strategy = MaxThroughputStrategy(backends=[TensorRTBackend(), TorchInductorJitBackend(), TorchEagerBackend()])
```
