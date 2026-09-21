---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Save and Load Checkpoints"
---

Use an AITune checkpoint when Python code will construct your model and load its tuned modules. The same save/load
APIs apply to an AITune Dynamo worker and to a Triton Python backend.

A checkpoint contains tuned module state and backend artifacts. Keep the application's model construction code and
dependencies available in the serving environment.

## Save after tuning

Save the model or pipeline after tuning completes:

```python
import aitune.torch as ait

ait.save(model, "model.ait")
```

The default storage directory is `checkpoints/`, so this writes `checkpoints/model.ait`. Pass the filename without
adding that directory again.

By default, saving also retains the unpacked `checkpoints/model/` directory and writes checksum information.
When deploying to another machine or container, make the checkpoint available there with the model code and the
dependencies required by its tuned backends.

## Load before serving

Construct the same model and load its tuned state. Replace `YourModel` with the constructor used by your application:

```python
import aitune.torch as ait

model = YourModel()
model.eval()
model.to("cuda")

model = ait.load(model, "model.ait")
output = model(input_data)
```

In a Dynamo worker, do this before calling `dynamo_worker()` or in its `setup` callback. In Triton's Python backend,
do it in `TritonPythonModel.initialize()`.

Loading extracts the checkpoint if a valid unpacked copy is unavailable, verifies checksums, and loads the saved
module state and backend artifacts. Later loads can reuse the unpacked files. Allow write access to the storage
directory when extraction is needed.

## Use another storage directory

Pass the same storage directory when saving and loading:

```python
from aitune.torch import LocalTorchStorage
import aitune.torch as ait

storage = LocalTorchStorage(base_folder="production/models")

# After tuning:
ait.save(model, "model.ait", storage=storage)

# In the serving application, after constructing the model:
model = ait.load(model, "model.ait", storage=storage)
```

Both calls resolve the filename to `production/models/model.ait`. Relative storage directories are resolved from
the process's working directory. In a container, use the directory where the checkpoint is mounted.

## Continue with deployment

- [Deploy with Dynamo](dynamo.md)
- [Use Triton's Python backend](triton.md#use-a-python-model-instead)
