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

# Nemo ASR Parakeet CTC 0.6B Pipeline Tuning with NVIDIA AITune

This example demonstrates how to use NVIDIA AITune to tune the Nemo ASR with Parakeet-CTC-0.6B model.

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

### Tuning and inference the model

To tune the ASR model, run:

```bash
tune --audio_path 2086-149220-0033.wav
```

To infer the ASR model, run:

```bash
inference --audio_path 2086-149220-0033.wav
```

To benchmark the ASR model, run

```bash
benchmark --audio_path 2086-149220-0033.wav
```

Note: if `--audio_path` is omitted the default audio file, bundled with this example, will be used.


### AI Dynamo ParakeetCTC Deployment

To run ParakeetCTC as AI Dynamo service, we have prepared a few additional configs and scripts.

Code starts in `parakeet_ctc/dynamo/backend.py`. Docker and Docker Compose are used to make setup simple.

First, start all services by running `docker compose --profile all up --detach`. This will build and start all required services.

After successful tuning and services start run below command to test the service.

```sh
python -m parakeet_ctc.dynamo.client --help # to see the prompts
python -m parakeet_ctc.dynamo.client --num-requests 1
python -m parakeet_ctc.dynamo.client --num-requests 2
python -m parakeet_ctc.dynamo.client --num-requests 4
python -m parakeet_ctc.dynamo.client --num-requests 8
python -m parakeet_ctc.dynamo.client --num-requests 100
```

Finally, to shut it down use `docker compose --profile all down`.

#### Dynamic batching

The service uses dynamic batching — requests are grouped and processed together for efficiency. Currently, there is one frontend and one worker. To support multiple workers, move batching to a separate service that handles request grouping.


## Model Details

Can be found in following pages:
* https://huggingface.co/nvidia/parakeet-ctc-0.6b
* https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/asr/models.html