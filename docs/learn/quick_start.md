<!--
Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

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
# Quick Start

This quick start provides examples of tuning and deployment paths available in NVIDIA AITune.

NVIDIA AITune enables seamless tuning of models for deployment (for example, converting them to TensorRT) without requiring changes to your original Python pipelines.

NVIDIA AITune supports two modes:

* Ahead-of-time tuning — provide a model or a pipeline, and a dataset/dataloader. You can either rely on `inspect` to detect promising modules to tune or manually select them.
* Just-in-time tuning — set a special environment variable, run your script without changes, and AITune will, on the fly, detect modules and tune them one by one.

Ahead-of-time mode is more powerful and allows you to tweak more settings, whereas just-in-time works out of the box but offers less control over the tuning process. For a more detailed comparison, see the [Comparison between AOT and JIT tuning](#comparison-between-ahead-of-time-and-just-in-time-tuning) section.

## Ahead-of-time tuning

The code below demonstrates Stable Diffusion pipeline tuning.

You can annotate `torch.nn.Module`s manually or use the `inspect` functionality to have modules picked automatically; you can then verify them and schedule them for tuning.

First, install the required third-party dependencies:

```bash
pip install transformers diffusers torch
```

Then initialize the pipeline:

```python
import aitune.torch as ait
from diffusers import DiffusionPipeline

# Initialize pipeline
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers")
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
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers")
pipe.to("cuda")
ait.load(pipe, "tuned_pipe.ait")
```

## Just-in-time tuning

In this mode, there is no need to modify the user's code. At the beginning, AITune uses a few inferences to detect model architecture and hierarchy of a model. Then it tries to tune modules one by one starting from the top. If there is one of the following conditions:

* a graph break is detected, i.e., torch.nn.Module contains conditional logic on inputs, meaning there is no guarantee of a static, correct graph of computations, or
* there is an error during tuning

that module is left unchanged and AITune tries to tune its children. This process continues until the module depth reaches a configured limit.

To turn on this mode, just set the following environment variable:

```bash
export AUTOWRAPT_BOOTSTRAP=aitune_enable_jit_tuning
```

You can then run your script without modifying it, for example:

```bash
python your_script.py
```

### Configuring just-in-time tuning

If there is a need to adjust just-in-time options, you can do it but currently this requires modifying code to import the JIT config:

```python
from aitune.torch.jit.config import config
from aitune.torch.backend import TensorRTBackend

config.max_depth_level = 1 # change the default maximum depth level for nested modules to be tuned
config.detect_graph_breaks = False # turn off graph break detection
config.backends = [TensorRTBackend()] # change the backends
```

## Comparison between ahead-of-time and just-in-time tuning

The ahead-of-time tuning gives you the most control over the tuning process:

* it detects the batch axis and dynamic axes (axes that change shape independently of batch size, e.g., sequence length in LLMs)
* allows picking modules to tune
* you can pick a tuning strategy (e.g., best throughput) for the whole process or per-module
* you can pick tuning backends (e.g., TensorRT, TorchInductor, TorchAO) which will be used by the strategy
* you can mix different backends in the same model/pipeline
* you can manually verify the tuning process (note: AITune performs basic checks for NaNs and errors)
* you can save the resulting artifact and later read it from disk

The big advantage of just-in-time tuning is that you don't need to modify the user's script to tune a model. However, it has some disadvantages - since it cannot access data directly (you don't provide a dataloader):

* it cannot deduce batch size nor do benchmarking
* input/output shapes depend on the data seen, so for example, TRT backend will build a profile only for that data
* it needs at least two inference calls - first to get model/pipeline hierarchy and second one for actual tuning
* if you need dynamic axes (e.g., TRT backend), you need to provide two different batch sizes
* there is limited support of strategies due to unknown batch size
* you can specify backends for the whole model

The following table summarizes the difference between modes:

| Feature                 | Ahead-of-time         | Just-in-time                  |
|-------------------------|-----------------------|-------------------------------|
| Detecting dynamic axes  | Yes                   | Yes                           |
| Extrapolating batches   | Yes                   | No                            |
| Benchmarking            | Yes                   | No (no extrapolating batches) |
| Modules for tuning      | User has full control | Picked automatically          |
| Selecting tune strategy | Global or per module  | Global                        |
| Available strategies    | All                   | Limited (no benchmarking)     |
| Tune time               | Slow                  | Quick                         |
| Saving artifacts        | Yes                   | No                            |
| Load tuned model time   | Quick                 | Re-tuning required            |
| Code changes required   | Yes                   | No                            |
| Caching                 | Yes                   | No                            |

Note: Currently, JIT mode does not support caching results, i.e., every time a new Python interpreter starts, the tuning process starts from scratch.