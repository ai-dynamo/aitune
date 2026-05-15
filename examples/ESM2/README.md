<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ESM2 Model Tuning with NVIDIA AITune

> Evolutionary-scale prediction of atomic level protein structure with a language model

This example demonstrates how to use NVIDIA AITune to tune the ESM2 transformer protein language model - `facebook/esm2_t33_650M_UR50D` - from Hugging Face's transformer library.

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

To tune the ESM2 model, run:

```bash
tune
```

The example saves the tuned checkpoint under `checkpoints/esm2_tuned.ait` and then copies the archive plus SHA sidecar to `/tmp/esm2_tuned.ait`.

After tuning, run inference

```bash
inference
```

`inference` loads the relocated checkpoint from `/tmp/esm2_tuned.ait`.

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
