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

# ResNet models tuning with NVIDIA AITune

This example demonstrates how to use NVIDIA AITune to tune a ResNet model.

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

### AI Dynamo Resnet Deployment

To run Resnet as AI Dynamo service, we have prepared a few additional configs and scripts.

Code starts in `resnet/dynamo/service.py`, Docker and Docker Compose is used to make setup simple.

Firstly, start all services by running `docker compose up --detach`. This will build and start all required services.

After successful tunning and services start run below command to test the service.

```sh
python -m resnet.dynamo.client --image-path dog.webp
# response should be:
#    {"prediction":"golden retriever","confidence":0.9992710947990417,"class_id":207}
```

Finally, to shut it down use `docker compose down`.

