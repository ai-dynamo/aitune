<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Flux Pipeline Tuning with NVIDIA AITune

This example demonstrates how to use NVIDIA AITune to tune the Flux text-to-image model from Hugging Face's diffusers library.

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

To tune the Flux model, run:

```bash
tune --model-name black-forest-labs/FLUX.1-dev --prompt "A futuristic cityscape with neon lights"
```

You can customize the following parameters:
- `--model-name`: HuggingFace model name or path (default: "black-forest-labs/FLUX.1-dev")
- `--prompt`: Text prompt for image generation
- `--negative-prompt`: Negative text prompt (default: "low quality, blurry")
- `--height`: Height of the generated image (default: 512)
- `--width`: Width of the generated image (default: 512)
- `--steps`: Number of inference steps (default: 25)

### Generating images with the tuned model

After tuning, generate images with:

```bash
inference --prompt "A beautiful landscape with mountains and a lake" --output-dir output
```

The generated image will be saved in the specified output directory.

### AI Dynamo FLUX Deployment

Serves the tuned FLUX model as an OpenAI-compatible image generation endpoint via [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo).

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
python -m flux.dynamo.client --prompt "A futuristic cityscape at night"
# Image saved to output.png
```

Or use any OpenAI-compatible client pointed at `http://localhost:8000/v1` with model `black-forest-labs/FLUX.1-dev`.

#### Dynamic batching

The service uses dynamic batching — requests are grouped and processed together for efficiency. Currently, there is one frontend and one worker. To support multiple workers, move batching to a separate service that handles request grouping.


## Model Details

The Flux model is a text-to-image diffusion model that generates high-quality images from text descriptions. The model is trained on a large dataset of images and text, and can generate realistic images across various domains.

For more information, visit the [Flux model page on HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-dev).

