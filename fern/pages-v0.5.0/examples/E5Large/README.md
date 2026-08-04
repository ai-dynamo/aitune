---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "E5 Large V2 Embedding"
---

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

### Logging hardware metrics

If you would like to log hardware metrics during tuning or inference, export `AITUNE_HARDWARE_METRICS=True` environment variable, e.g.

```bash
AITUNE_HARDWARE_METRICS=True uv run inference --prompt "query: What is the capital city of France?"
```

### Command-Line Options

- `--model-name`: SentenceTransformer model name (default: "intfloat/e5-large-v2")
- `--tuned-model-path`: Path to save/load the tuned model (default: "e5large_tuned.pt")
- `--prompt`: Text prompt for embedding (default: "query: how much protein should a female eat")
- `--max-batch-size`: Maximum batch size (default: 4)

### AI Dynamo E5Large

Serves the tuned E5Large model as an OpenAI-compatible embedding endpoint via [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo).

**Prerequisite:** tune the model first and set `Backend.tuned_model_path` in `config.yaml`.

`run_dynamo.sh` starts everything in one command — it launches the Dynamo HTTP frontend and the backend worker, waits for both to be ready, then runs a smoke-test embedding request:

```bash
uv pip install ".[dynamo]"
tune
./run_dynamo.sh
# Starting the frontend...
# Starting the backend...
# Waiting for dyn://aitune.backend.generate to appear in /health...
# Embedding dim: 1024
# First 5 values: [0.022, -0.034, ...]
```

## Model Details

Can be found in following pages:
* https://huggingface.co/intfloat/e5-large-v2
