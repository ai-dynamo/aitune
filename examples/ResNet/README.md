---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "ResNet models tuning"
---

This example demonstrates how to use NVIDIA AITune to tune a ResNet model.

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

To tune the ResNet model, run:

```bash
tune --model-name resnet50
```

or for uv:

```bash
uv run tune --model-name resnet50
```

After tuning, run classification with:

```bash
inference --model-name resnet50 --image-path your_image
```

or for uv:

```bash
uv run inference --model-name resnet50 --image-path your_image
```

### Logging hardware metrics

If you would like to log hardware metrics during tuning or inference export `AITUNE_HARDWARE_METRICS=True` envorinment variable e.g.

```bash
AITUNE_HARDWARE_METRICS=True uv run tune
```

### AI Dynamo ResNet Deployment with Batching

To run ResNet as AI Dynamo service with dynamic batching, we have prepared additional configs and scripts.

Code starts in `resnet/dynamo/backend.py`, Docker and Docker Compose is used to make setup simple.

Firstly, start all services by running `docker compose --profile all up --detach`. This will build and start all required services.

After successful tuning and services start run below commands to test the service.

#### Single Request
```sh
python -m resnet.dynamo.client --num-requests 1
# response should be:
#    {"request_id":"img-resnet-single","prediction":"golden retriever","confidence":0.9992710947990417,"class_id":207,"inference_time":0.123}
```

#### Batching Demonstration
```sh
python -m resnet.dynamo.client --num-requests 2
python -m resnet.dynamo.client --num-requests 4
python -m resnet.dynamo.client --num-requests 8
python -m resnet.dynamo.client --num-requests 100
```

#### Custom Image
```sh
python -m resnet.dynamo.client --image-path your_image.jpg --num-requests 4
```

Finally, to shut it down use `docker compose down --volumes`.

#### Dynamic batching

The service uses dynamic batching — requests are grouped and processed together for efficiency. Currently, there is one frontend and one worker. To support multiple workers, move batching to a separate service that handles request grouping.

## Model Details

Can be found in following pages:
* https://pytorch.org/vision/stable/models.html#classification
* https://huggingface.co/timm

