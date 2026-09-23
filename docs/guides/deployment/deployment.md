---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Deployment Overview"
---

Deploy your AITune-tuned model with NVIDIA Dynamo. Choose a Python worker to serve your model or pipeline, or use
Triton within Dynamo to serve an optimized artifact. Standalone Triton is also supported.

## Deploy with Dynamo

Dynamo provides the serving frontend, service discovery, and request routing. There are two ways to connect your
tuned model:

### Serve a Python model or pipeline

Start here when you want to keep your existing Python inference code. Your application constructs the model and
loads an AITune checkpoint. AITune's Dynamo worker exposes your inference function through the OpenAI-compatible
HTTP frontend. Start with FLUX image generation or WAN video generation; audio and embeddings are also supported.

Follow [Deploy with Dynamo](dynamo.md) to serve FLUX on one GPU and generate your first image. Continue with its
[multi-GPU section](dynamo.md#multi-gpu-workers) for WAN video generation across multiple GPUs.

### Serve a model through Triton within Dynamo

Use this path when you want Triton to execute the model behind Dynamo's frontend and routing. AITune can generate a
Triton repository for an ONNX Runtime, TensorRT, or TorchInductor AOT artifact. Requests use the KServe gRPC tensor API.

Follow the [Triton guide](triton.md) to prepare the repository, then choose
[Run Triton through Dynamo](triton.md#run-triton-through-dynamo). That guide also explains how to use a checkpoint
with Triton's Python backend when your model needs Python code.

## Alternative: standalone Triton

If you already deploy with Triton or want to serve directly from it, use the same model repository with
[standalone Triton](triton.md#run-triton-standalone). You can either generate a repository for an optimized artifact
or provide a Python backend that loads an AITune checkpoint. Both are covered in the [Triton guide](triton.md).

## What to prepare

The serving platform and the saved model format are separate choices:

| What runs inference | What you prepare | Where it can run |
|---|---|---|
| Python code that constructs your model or pipeline | An AITune checkpoint, model code, and dependencies | AITune's Dynamo worker or Triton's Python backend |
| Triton loading an optimized artifact directly | A generated Triton model repository | Dynamo's Triton worker or standalone Triton |

The [save and load guide](checkpoints.md) applies to both Python deployment paths. Repository generation is covered
in the [Triton guide](triton.md#generate-a-model-repository). Each deployment guide links to the preparation steps it
needs.
