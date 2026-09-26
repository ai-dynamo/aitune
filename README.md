<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA AITune

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org/)

**NVIDIA AITune automates the acceleration of PyTorch models and pipelines on NVIDIA GPUs.**
It brings inference engines and acceleration techniques together under a single API to find a suitable
configuration for each workload.

Start with a model from Hugging Face or timm, or bring your own model and checkpoint. For supported inference
workloads, enable just-in-time tuning with **zero changes to your application code**, or use the explicit tuning
API to prepare artifacts for deployment.

Find your model in the [recipe catalog](examples/README.md), or follow the [quick start](docs/learn/quick_start.md).

## Why AITune?

- **Reduce manual integration work.** Automate module inspection, backend evaluation, numerical checks, and
  performance-based selection through one API.
- **Explore multiple acceleration paths.** Different backends perform best on different models and workloads.
  AITune evaluates configured, compatible candidates and can select different implementations for different modules.
- **Match backends to each module.** AITune resolves compatible backend candidates for each module and its execution
  context, allowing different parts of a pipeline to use different acceleration paths.
- **Tune to your requirements.** Set performance goals and compilation requirements. AITune automatically resolves
  and evaluates suitable backends for each module.
- **Automate deployment preparation.** Generate Triton model stores or serve tuned artifacts through Dynamo workers.

## How it works

AITune combines these stages into a workflow for each model or pipeline:

1. **Inspect.** Provide a PyTorch model and representative inputs. AITune finds tunable `nn.Module` components
   and observes their inputs and execution.
2. **Wrap.** Customize the selected modules' tuning strategies and backend configurations.
3. **Tune.** Try compatible backend candidates and acceleration combinations, including post-training quantization
   (PTQ), kernel selection, and CUDA graphs. Check numerical outputs against the original implementation, measure
   performance, and select a backend using the configured strategy.
4. **Run & deploy.** Run tuned modules in Python. Use the explicit workflow to save an `.ait` artifact for reuse
   or deployment with Triton or Dynamo, as supported by the recipe.

<a href="docs/assets/aitune_workflow.svg">
  <img src="docs/assets/aitune_workflow.svg" alt="AITune workflow: inspect, wrap, try and assess backends, select the best backend, and run or deploy." width="800" />
</a>

## Install

Use a Linux environment with an NVIDIA GPU and a compatible PyTorch and CUDA installation:

```bash
pip install --extra-index-url https://pypi.nvidia.com aitune
```

See the [installation guide](docs/learn/install.md) for requirements, containers, and PyTorch/CUDA version selection.

## Quick start

Both examples use Stable Diffusion. Install the model dependencies:

```bash
pip install diffusers transformers
```

### Just-in-time (JIT): try acceleration in your existing application

Just-in-time tuning happens as your application runs, using inputs captured during inference.
Save this as `your_script.py`:

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

image = pipe("A landscape with mountains and a lake").images[0]
image.save("landscape.png")
```

Enable AITune when launching the script:

```bash
AUTOWRAPT_BOOTSTRAP=aitune_enable_jit_tuning python your_script.py
```

The example saves an image to `landscape.png`. Use representative prompts for your workload.
Eligible modules use the selected implementations in the running application.
A new process starts tuning again; use the explicit workflow below for saved artifacts.
See the [JIT guide](docs/guides/jit_tuning.md) for configuration and deferred tuning for pipelines.

### Ahead-of-time (AOT): prepare a reusable tuning artifact

Ahead-of-time tuning prepares your model for deployment with a consistent flow:

<p align="center">
  <strong>inspect → wrap → tune → deploy</strong>
</p>

Provide your model and representative inputs:

```python
import torch
from diffusers import DiffusionPipeline

import aitune.torch as ait

pipe = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")
inputs = [{"prompt": "A landscape with mountains and a lake"}]

modules = ait.inspect(pipe, inputs)
pipe = ait.wrap(pipe, modules.get_modules())
ait.tune(pipe, inputs)

image = pipe(**inputs[0]).images[0]
image.save("landscape.png")
ait.save(pipe, "tuned_pipe.ait")
```

This produces an image and a `tuned_pipe.ait` artifact that can be loaded into a compatible pipeline.
Use representative prompts or data for your workload. See the [AOT guide](docs/guides/aot_tuning.md) for tuning
configuration and the [checkpoint guide](docs/guides/deployment/checkpoints.md) for saving and loading.

## Model recipes

Explore [ready-to-use model recipes](examples/README.md) for tuning, validation, benchmarking, and deployment.
Choose a model and a configuration for your GPU setup, precision, and deployment target, then adapt the recipe
with your own compatible checkpoint and data.

<!-- Add headline speedups here when measured recipe results are available. Link each claim to its workload,
     hardware, and comparisons against the Original Model. Document engineering effort with concrete integration
     tasks and reproducible setup steps; quantify time or cost savings only when supporting measurements exist. -->

## Learn more

- [Product overview](docs/learn/overview.md)
- [Supported backends](docs/learn/backends.md)
- [Multi-GPU tuning](docs/guides/multi_gpu.md)
- [Profiling and hardware metrics](docs/guides/advanced/profiling_and_hardware_metrics.md)
- [Deployment overview](docs/guides/deployment/deployment.md)
- [Triton deployment](docs/guides/deployment/triton.md)
- [Dynamo deployment](docs/guides/deployment/dynamo.md)

## Notice

**NOTICE AND DISCLAIMER: This software automatically retrieves, accesses or interacts with external materials. Those    retrieved materials are not distributed with this software and are governed solely by separate terms, conditions and licenses.  You are solely responsible for finding, reviewing and complying with all applicable terms, conditions, and licenses, and for verifying the security, integrity and suitability of any retrieved materials for your specific use case. This software is provided "AS IS", without warranty of any kind. The author makes no representations or warranties regarding any retrieved materials, and assumes no liability for any losses, damages, liabilities or legal consequences from your use or inability to use this software or any retrieved materials. Use this software and the retrieved materials at your own risk.**
