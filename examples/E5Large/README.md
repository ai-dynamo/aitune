<!--
Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

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

# E5 Large V2 Embedding

This example demonstrates how to use NVIDIA AI Tune to optimize the HuggingFace E5Large v2 embeddings.

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

### Tuning and inference the model

To optimize the embedding model, run:

```bash
tune
```

To infer the embedding model, run:

```bash
inference --prompt "query: What is the capital city of France?"
```

### Command-Line Options

- `--model-name`: SentenceTransformer model name (default: "intfloat/e5-large-v2")
- `--tuned-model-path`: Path to save/load the tuned model (default: "e5large_tuned.pt")
- `--prompt`: Text prompt for embedding (default: "query: how much protein should a female eat")
- `--max-batch-size`: Maximum batch size (default: 4)

### AI Dynamo E5Large Deployment

To run E5Large as AI Dynamo service, we have prepared a few additional configs and scripts.

The service is split into backend (`e5large/dynamo/backend.py`) and frontend (`e5large/dynamo/frontend.py`) components. Docker and Docker Compose are used to make setup simple.

First, start all services by running `HF_TOKEN=hf.... docker compose --profile all up --detach`. This will build and start all required services. The token for the HuggingFace is required to download the model.

After successful download, tuning and services start run below command to test the service.

```sh
python -m e5large.dynamo.client --help # to see the prompts
python -m e5large.dynamo.client --num-requests 1
python -m e5large.dynamo.client --num-requests 2
python -m e5large.dynamo.client --num-requests 4
python -m e5large.dynamo.client --num-requests 8
python -m e5large.dynamo.client --num-requests 100
```

Finally, to shut it down use `docker compose --profile all down`.

#### Dynamic batching

The service uses dynamic batching — requests are grouped and processed together for efficiency. Currently, there is one frontend and one worker. To support multiple workers, move batching to a separate service that handles request grouping.


## Model Details

Can be found in following pages:
* https://huggingface.co/intfloat/e5-large-v2
