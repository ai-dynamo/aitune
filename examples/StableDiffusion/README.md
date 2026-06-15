---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Stable Diffusion Pipeline Tuning"
---

This example demonstrates how to use NVIDIA AITune to tune the Stable Diffusion text-to-image model from Hugging Face's diffusers library.

## Environment Setup

You can use either of the following options to set up the environment:

### Option 1 - virtual environment managed by you

Activate your virtual environment and install the dependencies:

```bash
pip install --extra-index-url https://pypi.nvidia.com .
```

### Option 2 - virtual environment managed by `uv`

Install dependencies:

```bash
uv sync
```

## Usage

### Tuning the model

To tune the Stable Diffusion model, run:

```bash
tune --model-name stabilityai/stable-diffusion-3-medium-diffusers --prompt "A futuristic cityscape with neon lights"
```

You can customize the following parameters:
- `--model-name`: HuggingFace model name or path (default: "stabilityai/stable-diffusion-3-medium-diffusers")
- `--prompt`: Text prompt for image generation
- `--sizes`: Space-separated `width,height` image sizes (default: `512,512 1024,1024`)
- `--steps`: Number of inference steps (default: 50)
- `--tuned-model-path`: Path to save or load the tuned model (default: `stable_diffusion.ait`)

### Generating images with the tuned model

After tuning, generate images with:

```bash
AITUNE_OUTPUT_DIR=output inference --prompt "A beautiful landscape with mountains and a lake"
```

The generated image will be saved in `AITUNE_OUTPUT_DIR`, or `output` when the environment variable is not set.


### Logging hardware metrics

If you would like to log hardware metrics during tuning or inference, export `AITUNE_HARDWARE_METRICS=True` environment variable, e.g.

```bash
AITUNE_HARDWARE_METRICS=True uv run inference --prompt "A beautiful landscape with mountains and a lake"
```

### AI Dynamo Stable Diffusion Deployment

Serves the tuned Stable Diffusion model as an OpenAI-compatible image generation endpoint via [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo).

**Prerequisite:** tune the model first and set `Backend.tuned_model_path` in `config.yaml`.

`run_dynamo.sh` starts everything in one command — it launches the Dynamo HTTP frontend and the backend worker, waits for both to be ready, then runs a smoke-test image generation request:

```bash
./run_dynamo.sh
# Starting the frontend...
# Starting the backend...
# Waiting for dyn://aitune.backend.generate to appear in /health...
# Image saved to output.png
```

The frontend listens on port 8000 (OpenAI-compatible). You can also call it directly:

```bash
python -m stable_diffusion.dynamo.client --prompt "A serene mountain landscape at sunset"
# Image saved to output.png
```

Or use any OpenAI-compatible client pointed at `http://localhost:8000/v1` with model `stabilityai/stable-diffusion-3-medium-diffusers`.

#### Dynamic batching

The service uses dynamic batching — requests are grouped and processed together for efficiency. Currently, there is one frontend and one worker. To support multiple workers, move batching to a separate service that handles request grouping.

## Model Details

The Stable Diffusion model is a text-to-image diffusion model that generates high-quality images from text descriptions. The model is trained on a large dataset of images and text, and can generate realistic images across various domains.

For more information, visit the [Stable Diffusion model page on HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers).
