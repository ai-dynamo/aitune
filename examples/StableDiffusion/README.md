<!--
Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

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

# Flux Pipeline Tuning with NVIDIA AITune

This example demonstrates how to use NVIDIA AITune to tune the Stable Diffusion text-to-image model from Hugging Face's diffusers library.

## Environment Setup

You can use either of the following options to setup the environment:

### Option 1 - virtual environment managed by you

Activate your virtual environment and install the dependencies:

```bash
pip install .
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
tune --model-name stabilityai/stable-diffusion-2-1 --prompt "A futuristic cityscape with neon lights"
```

You can customize the following parameters:
- `--model-name`: HuggingFace model name or path (default: "stabilityai/stable-diffusion-2-1")
- `--prompt`: Text prompt for image generation
- `--negative-prompt`: Negative text prompt (default: "low quality, blurry")
- `--height`: Height of the generated image (default: 512)
- `--width`: Width of the generated image (default: 512)
- `--steps`: Number of inference steps (default: 50)

### Generating images with the tuned model

After tuning, generate images with:

```bash
inference --prompt "A beautiful landscape with mountains and a lake" --output-dir output
```

The generated image will be saved in the specified output directory.

## Model Details

The Stable Diffusion model is a text-to-image diffusion model that generates high-quality images from text descriptions. The model is trained on a large dataset of images and text, and can generate realistic images across various domains.

For more information, visit the [Stable Diffusion model page on HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-1).