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

This example demonstrates how to use NVIDIA AITune to tune the Flux text-to-image model from Hugging Face's diffusers library.

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

To run FLUX as AI Dynamo service, we have prepared a few additional configs and scripts.

The service is split into backend (`flux/dynamo/backend.py`) and frontend (`flux/dynamo/frontend.py`) components. Docker and Docker Compose is used to make setup simple.

Firstly, start all services by running `HF_TOKEN=hf.... docker compose --profile all up --detach`. This will build and start all required services. The token for the HuggingFace is required to download the model.

After successful download, tuning and services start run below command to test the service.

```sh
python -m flux.dynamo.client --help # to see the prompts
python -m flux.dynamo.client --num-requests 1
python -m flux.dynamo.client --num-requests 2
python -m flux.dynamo.client --num-requests 4
python -m flux.dynamo.client --num-requests 8
python -m flux.dynamo.client --num-requests 100
```

Finally, to shut it down use `docker compose --profile all down`.

#### Dynamic batching

The service uses dynamic batching — requests are grouped and processed together for efficiency. Currently, there is one frontend and one worker. To support multiple workers, move batching to a separate service that handles request grouping.


## Model Details

The Flux model is a text-to-image diffusion model that generates high-quality images from text descriptions. The model is trained on a large dataset of images and text, and can generate realistic images across various domains.

For more information, visit the [Flux model page on HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-dev).

