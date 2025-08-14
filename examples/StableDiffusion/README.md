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


### AI Dynamo Stable Diffusion Deployment

To run Stable Diffusion as AI Dynamo service, we have prepared a few additional configs and scripts.

Code starts in `stable_diffusion/dynamo/service.py`, Docker and Docker Compose is used to make setup simple.

Firstly, start all services by running `docker compose --profile all up --detach`. This will build and start all required services.

After successful tunning and services start run below command to test the service.

```sh
python -m stable_diffusion.dynamo.client --help # to see the prompts
python -m stable_diffusion.dynamo.client --num-requests 1
python -m stable_diffusion.dynamo.client --num-requests 2
python -m stable_diffusion.dynamo.client --num-requests 4
python -m stable_diffusion.dynamo.client --num-requests 8
python -m stable_diffusion.dynamo.client --num-requests 100
```

Finally, to shut it down use `docker compose --profile all down`.


#### Development version

```
cd examples/StableDiffusion

# 1. Start auxiliary services NATS and ETCD
docker compose --profile aux up -d

# 2. build docker image
docker build --build-context aitune=../.. -f Dockerfile.dynamo -t sdd .

# 3. Running docker with caches
mkdir -p tmp/hf tmp/ait

docker run \
    --rm -ti \
    --name sdd_dynamo_demo \
    --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  \
    --network stablediffusion_default \
    -v `pwd`/tmp/hf:/root/.cache/huggingface \
    -v `pwd`/tmp/ait:/ait_cache \
    sdd:latest bash

# 4. Running service (inside container)
. /opt/dynamo/venv/bin/activate
export ETCD_ENDPOINTS=http://etcd-server:2379
export NATS_SERVER=nats://nats-server:4222
export AITUNE_CACHE_DIR=/ait_cache

dynamo serve stable_diffusion.dynamo.service:StableDiffusionBatchedFrontend -f config.yaml

# ... first time it will tune for a while

# 5. Run client (in separate shell, same container)
docker exec -ti sdd_dynamo_demo bash
. /opt/dynamo/venv/bin/activate

# 5.a. optionally link generated images to temporary cache dir, for viewing
mkdir generated_images
ln -s ./generated_images /ait_cache/

python -m stable_diffusion.dynamo.client --num-requests 8
```

#### Dynamic batching

The service uses dynamic batching — requests are grouped and processed together for efficiency. Currently, there is one frontend and one worker. To support multiple workers, move batching to a separate service that handles request grouping.

## Model Details

The Stable Diffusion model is a text-to-image diffusion model that generates high-quality images from text descriptions. The model is trained on a large dataset of images and text, and can generate realistic images across various domains.

For more information, visit the [Stable Diffusion model page on HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-1).