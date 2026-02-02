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

# ESM2 Model Tuning with NVIDIA AITune

> Evolutionary-scale prediction of atomic level protein structure with a language model

This example demonstrates how to use NVIDIA AITune to tune the ESM2 transformer protein language model - `facebook/esm2_t33_650M_UR50D` - from Hugging Face's transformer library.

## Environment Setup

You can use either of the following options to set up the environment:

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

To tune the ESM2 model, run:

```bash
tune
```

After tuning, run inference

```bash
inference
```

### AI Dynamo ESM2 Deployment

To run ESM2 as AI Dynamo service, we have prepared a few additional configs and scripts.

The service is split into backend (`esm2/dynamo/backend.py`) and frontend (`esm2/dynamo/frontend.py`) components. Docker and Docker Compose are used to make setup simple.

First, start all services by running `HF_TOKEN=hf.... docker compose --profile all up --detach`. This will build and start all required services. The token for the HuggingFace is required to download the model.

After successful download, tuning and services start run below command to test the service.

```sh
python -m esm2.dynamo.client # same as with --num-requests 1
python -m esm2.dynamo.client --num-requests 2
python -m esm2.dynamo.client --num-requests 4
python -m esm2.dynamo.client --num-requests 8
python -m esm2.dynamo.client --num-requests 100
```

Finally, to shut it down use `docker compose --profile all down`.

#### Dynamic batching

The service uses dynamic batching — requests are grouped and processed together for efficiency. Currently, there is one frontend and one worker. To support multiple workers, move batching to a separate service that handles request grouping.


## Model Details

ESM-2 (Evolutionary Scale Modeling-2) is a state-of-the-art protein language model developed by Facebook AI, designed to analyze and interpret protein sequences using deep learning techniques. It is trained on a masked language modeling objective, meaning it predicts missing amino acids in protein sequences, which enables it to learn patterns relevant for understanding structure and function.

## Links

* [Hugging Face Model](https://huggingface.co/facebook/esm2_t33_650M_UR50D)
* [Research Paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2)