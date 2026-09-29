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

**NVIDIA AITune automates inference tuning for PyTorch models and pipelines on NVIDIA GPUs.**
It brings inference engines and acceleration techniques together under a single, extensible API. AITune combines
backend evaluation, numerical validation, and performance measurement into one workflow, selecting implementations
for individual modules according to the configured tuning strategy.

Start with a model from Hugging Face or timm, or bring your own model and checkpoint. For supported applications,
enable just-in-time tuning during inference, or use the explicit tuning API to prepare artifacts for deployment.

Find your model in the [recipe catalog](../../examples/README.md), or follow the [quick start](quick_start.md).

## Why AITune?

- **Integrate through one API.** Use a shared workflow across supported backends. Extend the workflow with custom
  backends and tuning strategies.
- **Evaluate options per module.** Use default candidates that account for execution needs, or configure candidates
  explicitly. Different modules in a pipeline can use different backends.
- **Select using measurements.** Check numerical outputs and measure performance on representative inputs. Choose
  strategies for throughput, latency, or a latency budget to guide selection.
- **Prepare for deployment.** Reuse tuned modules in Python, generate Triton model repositories, or serve through
  Dynamo workers, as supported by the recipe.

## How it works

AITune combines these stages into a workflow for each model or pipeline:

1. **Inspect.** Provide a PyTorch model and representative inputs. AITune finds tunable `nn.Module` components
   and observes their inputs and execution.
2. **Wrap.** Customize the selected modules' tuning strategies and backend configurations.
3. **Tune.** Evaluate configured backend candidates, check numerical outputs against the original implementation,
   and measure performance on representative inputs. Select a backend using the configured strategy. Candidates can
   incorporate post-training quantization (PTQ), kernel selection, or CUDA graphs where supported by the backend.
4. **Run & deploy.** Run tuned modules in Python. Use the explicit workflow to save an `.ait` artifact for reuse
   or deployment with Triton or Dynamo, as supported by the recipe.

<a href="../assets/aitune_workflow.svg">
  <img src="../assets/aitune_workflow.svg" alt="AITune workflow: inspect, wrap, evaluate backend candidates, validate outputs, measure performance, select the best backend for the configured strategy, and run or deploy." width="800" />
</a>

A shared tuning workflow brings supported backends and acceleration techniques together, with selection guided by
the configured strategy.

## Backend selection within a pipeline

Default candidates account for the tuning workflow and whether a module requires distributed execution. You can
also configure the candidates explicitly. For example, a video pipeline may contain a transformer that spans
multiple GPUs and a decoder that runs locally on each GPU. The transformer needs backends that support distributed
execution, while the decoder can use single-GPU candidates.

Each module can receive a different backend. Selection depends on the evaluated candidates, representative inputs,
hardware, and tuning strategy. Measure the complete pipeline with your workload to assess the overall result.
See [tuning strategies](../guides/tune_strategies/tune_strategies.md) for selection policies and
[multi-GPU tuning](../guides/multi_gpu.md) for distributed execution.

To add another backend or selection policy, see
[workflow customization](../guides/advanced/tuning_workflow.md#workflow-customization).

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
