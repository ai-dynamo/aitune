---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Quick Start"
---

This quick start provides examples of tuning and deployment paths available in NVIDIA AITune.

NVIDIA AITune enables seamless tuning of models for deployment (for example, converting them to TensorRT) without requiring changes to your original Python pipelines.

NVIDIA AITune supports two modes:

* Ahead-of-time tuning — provide a model or a pipeline, and a dataset/dataloader. You can either rely on `inspect` to detect promising modules to tune or manually select them.
* Just-in-time tuning — set a special environment variable, run your script without changes, and AITune will, on the fly, detect modules and tune them one by one.

Ahead-of-time mode is more powerful and allows you to tweak more settings, whereas just-in-time works out of the box but offers less control over the tuning process. For a more detailed comparison, see the [Comparison between AOT and JIT tuning](#comparison-between-ahead-of-time-and-just-in-time-tuning) section.

## Enabling logging

The tuning process guides the user through decisions and steps that are performed to tune every selected module.

We recommend to enable the INFO logging level for better verbosity in the quick start steps:
```python
import logging

logging.basicConfig(level=logging.INFO, force=True)
```

Learn about more options in [observability](observability.md).

## Ahead-of-time tuning

The code below demonstrates Stable Diffusion pipeline tuning.

You can annotate `torch.nn.Module`s manually or use the `inspect` functionality to have modules picked automatically; you can then verify them and schedule them for tuning.

First, install the required third-party dependencies:

```bash
pip install transformers diffusers torch
```

Then initialize the pipeline:

```python
import torch
from diffusers import DiffusionPipeline

import aitune.torch as ait

# Initialize pipeline
pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.to("cuda")
```

Next, `inspect` the pipeline components and display the summary:

```python
# Prepare input data
input_data = [{"prompt": "A beautiful landscape with mountains and a lake"}]

# Inspect pipeline to get modules
modules_info = ait.inspect(pipe, input_data)


# Optional: inference function, if you need more control over execution
def infer(prompt):
    return pipe(prompt, width=1024, height=1024, num_inference_steps=10)

# modules_info = ait.inspect(pipe, input_data, inference_function=infer)

# Display modules info
modules_info.describe()
```

Finally, `wrap` the selected modules and `tune` within the pipeline:

```python
# Wrap modules for tuning
modules = modules_info.get_modules()
pipe = ait.wrap(pipe, modules)

# Tune pipeline
ait.tune(pipe, input_data)
```

At this point, you can use the pipeline to generate predictions with the tuned models directly in Python:

```python
# Run inference on tuned pipeline
images = pipe(["A beautiful landscape with mountains and a lake"])
image = images[0][0]

# Save image for preview
image.save("landscape.png")
```

Once the pipeline has been tuned, you can save the best-performing version of the modules for later deployment:

```python
ait.save(pipe, "tuned_pipe.ait")
```

And load the tuned pipeline directly:

```python
pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.to("cuda")
ait.load(pipe, "tuned_pipe.ait")
```

## Just-in-time tuning

In this mode, there is no need to modify the user's code. AITune records inference calls until `jit_config.min_samples` are collected, then tries to tune modules one by one starting from the top. If there is one of the following conditions:

* a graph break is detected, i.e., torch.nn.Module contains conditional logic on inputs, meaning there is no guarantee of a static, correct graph of computations, or
* there is an error during tuning

that module is left unchanged and AITune tries to tune its children. This process continues until the module depth reaches a configured limit.

First, install the required third-party dependencies:

```bash
pip install transformers diffusers torch
```

Prepare the script with the model for tuning `my_script.py`:

```python
# Enable JIT tuning - single import
import aitune.torch.jit.enable

from diffusers import DiffusionPipeline

# Initialize pipeline
pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.to("cuda")

# First call - tuning the model
pipe("A beautiful landscape with mountains and a lake")

# Second call - using tuned model
pipe("A beautiful landscape with mountains and a lake")
```

You can then run your script:

```bash
python my_script.py
```

*Note*: The `import aitune.torch.jit.enable` must be a first import in your code. The alternative option is to use `export AUTOWRAPT_BOOTSTRAP=aitune_enable_jit_tuning` to avoid any source code modification.

### Configuring just-in-time tuning

If there is a need to adjust just-in-time options, you can do it but currently this requires modifying code to import the JIT config:

```python
from aitune.torch import jit_config
from aitune.torch.backend import TensorRTBackend
from aitune.torch.tune_strategy import FirstWinsStrategy

jit_config.max_depth_level = 2 # change the default maximum depth level for nested modules to be tuned
jit_config.detect_graph_breaks = False # turn off graph break detection
jit_config.strategy = FirstWinsStrategy(backends=[TensorRTBackend()]) # change the tune strategy
```

### Deferred mode

By default JIT tuning uses **eager mode** — each module is tuned automatically as soon as enough samples have been collected.  For pipelines where different modules are called a variable number of times per step (e.g. text-to-image or text-to-video diffusion models), use **deferred mode** instead. In deferred mode, AITune records samples until you mark a safe synchronization point; tuning then starts on the next normal forward pass.

```python
import aitune.torch.jit.enable

from aitune.torch import jit_config
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.tune import deferred as jit_deferred
from diffusers import DiffusionPipeline

jit_config.mode = JITMode.TUNE_DEFERRED

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers")
pipe.to("cuda")

# First full step — records samples for all modules
pipe("A beautiful landscape")

# Mark that deferred tuning may start
jit_deferred()

# This call triggers tuning from the normal pipeline flow
pipe("A snowy mountain at sunset")

# Subsequent calls run on the tuned pipeline
pipe("A snowy mountain at sunset")
```

See [JIT Tuning Guide](../guides/jit_tuning.md#tuning-modes) for a detailed comparison of the two modes.

## Comparison between ahead-of-time and just-in-time tuning

The ahead-of-time tuning gives you the most control over the tuning process:

* it detects the batch axis and dynamic axes (axes that change shape independently of batch size, e.g., sequence length in LLMs)
* allows picking modules to tune
* you can pick a tuning strategy (e.g., best throughput) for the whole process or per-module
* you can pick tuning backends (e.g., TensorRT, TorchInductor, TorchAO, ONNXRuntime) which will be used by the strategy
* you can mix different backends in the same model/pipeline
* you can manually verify the tuning process (note: AITune performs basic checks for NaNs and errors)
* you can save the resulting artifact and later read it from disk

The big advantage of just-in-time tuning is that you don't need to modify the user's script to tune a model. However, it has some disadvantages - since it cannot access data directly (you don't provide a dataloader):

* it cannot deduce batch size nor do benchmarking
* input/output shapes depend on the data seen, so for example, TRT backend will build a profile only for that data
* it needs at least one inference call to record inputs before tuning; later calls use tuned modules where tuning succeeded
* if you need dynamic axes (e.g., TRT backend), you need to provide two different batch sizes
* benchmarking-based strategies are limited because JIT cannot extrapolate to controlled batch sizes
* you can specify a global tune strategy for the whole model

The following table summarizes the difference between modes:

| Feature                 | Ahead-of-time         | Just-in-time                  |
|-------------------------|-----------------------|-------------------------------|
| Detecting dynamic axes  | Yes                   | Yes                           |
| Extrapolating batches   | Yes                   | No                            |
| Benchmarking            | Yes                   | No (no extrapolating batches) |
| Modules for tuning      | User has full control | Picked automatically          |
| Selecting tune strategy | Global or per module  | Global                        |
| Available strategies    | All                   | Global only                   |
| Tune time               | Slow                  | Quick                         |
| Saving artifacts        | Yes                   | No                            |
| Load tuned model time   | Quick                 | Re-tuning required            |
| Code changes required   | Yes                   | No                            |
| Caching                 | Yes                   | Build artifacts only          |

Note: JIT mode writes build artifacts and logs under `jit_config.cache_dir` / `AITUNE_JIT_CACHE_DIR`, but it does not reuse them as tuned checkpoints across Python interpreter runs. Every new process starts tuning from scratch.
