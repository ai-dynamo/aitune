---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Nemo ASR Parakeet RNNT 1.1B Pipeline Tuning"
---

This example demonstrates how to use NVIDIA AITune to tune the Nemo ASR with Parakeet RNNT 1.1B model.

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

### Sample audio file

The example uses a sample audio file that is **downloaded automatically** when you run the
commands below without an explicit `--audio_path`.
You can also download it manually:

```bash
wget https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav
```

### Tuning and inference the model

To tune the ASR model, run:

```bash
tune
```

To infer the ASR model, run:

```bash
inference
```


### Logging hardware metrics

If you would like to log hardware metrics during tuning or inference, export `AITUNE_HARDWARE_METRICS=True` environment variable, e.g.

```bash
AITUNE_HARDWARE_METRICS=True uv run inference
```

### AI Dynamo ParakeetRNNT Deployment

To run ParakeetRNNT as an AI Dynamo service, you need to first tune your model and then launch a test run using the provided script.

```sh
uv pip install ".[dynamo]"
tune
./run_dynamo.sh
```

This script starts the frontend and backend services, waits for them to be ready, then runs a test client to send sample requests. Once the test completes, all services are automatically shut down. This is meant as a functional check, not to provide a permanent server.

#### Dynamic batching

The service uses dynamic batching — requests are grouped and processed together for efficiency. Currently, there is one frontend and one worker. To support multiple workers, move batching to a separate service that handles request grouping.

## Model Details

Can be found in following pages:
* https://huggingface.co/nvidia/parakeet-rnnt-1.1b
* https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/asr/models.html