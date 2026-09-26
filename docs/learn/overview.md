---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "NVIDIA AITune"
---

import { BadgeLinks } from "../_components/BadgeLinks";

<BadgeLinks
  badges={[
    {
      href: "https://github.com/ai-dynamo/aitune/blob/main/LICENSE",
      src: "https://img.shields.io/badge/License-Apache_2.0-blue",
      alt: "License",
    },
    {
      href: "https://www.python.org/downloads/",
      src: "https://img.shields.io/badge/python-3.11%2B-blue",
      alt: "Python",
    },
    {
      href: "https://pytorch.org/",
      src: "https://img.shields.io/badge/PyTorch-2.8%2B-red",
      alt: "PyTorch",
    },
  ]}
/>

**NVIDIA AITune automates the acceleration of PyTorch models and pipelines on NVIDIA GPUs.**
It brings inference engines and acceleration techniques together under a single API to find a suitable
configuration for each workload.

Start with a model from Hugging Face or timm, or bring your own model and checkpoint. For supported inference
workloads, enable just-in-time tuning with **zero changes to your application code**, or use the explicit tuning
API to prepare artifacts for deployment.

Find your model in the [recipe catalog](../../examples/README.md), or follow the [quick start](quick_start.md).

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

<a href="../assets/aitune_workflow.svg">
  <img src="../assets/aitune_workflow.svg" alt="AITune workflow: inspect, wrap, try and assess backends, select the best backend, and run or deploy." width="800" />
</a>

## Getting started

- **Just-in-time (JIT): try acceleration in your existing application.** Enable tuning when launching your script;
  AITune captures inputs and tunes eligible modules during inference. See the [JIT guide](../guides/jit_tuning.md).
- **Ahead-of-time (AOT): prepare a reusable tuning artifact.** Use the explicit flow to inspect, wrap, and tune your model,
  then save it for reuse or deployment. See the [AOT guide](../guides/aot_tuning.md).

Follow the [quick start](quick_start.md) for a Stable Diffusion example and the [installation guide](install.md)
for environment setup.

## Model recipes

Explore [ready-to-use model recipes](../../examples/README.md) for tuning, validation, benchmarking, and deployment.
Choose a model and a configuration for your GPU setup, precision, and deployment target, then adapt the recipe
with your own compatible checkpoint and data.
