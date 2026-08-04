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

### User-provided dynamic shapes

The dynamic-shapes mode uses the same tuning flow while explicitly defining its batch and spatial ranges. With the
defaults, it records batch sizes 1–4 at `224×224` and then runs inference at the unseen shape `(2, 3, 256, 256)`:

```bash
uv run tune --dynamic-shapes 1
uv run inference --dynamic-shapes 1
```

The full-rank shape definition uses integers for fixed dimensions. Dimensions with the same name are shared, even
when defined as separate objects:

```python
from aitune.torch import BatchDim, DynamicDim

batch = BatchDim("batch", min=1, opt=4, max=4)
height = DynamicDim("spatial", min=224, opt=224, max=256)
width = DynamicDim("spatial", min=224, opt=224, max=256)
dynamic_shapes = {"x": (batch, 3, height, width)}
```

### Logging hardware metrics

If you would like to log hardware metrics during tuning or inference export `AITUNE_HARDWARE_METRICS=True` envorinment variable e.g.

```bash
AITUNE_HARDWARE_METRICS=True uv run tune
```

### AI Dynamo ResNet Deployment with Batching

Run ResNet as an AI Dynamo service with dynamic batching:

```bash
pip install ".[dynamo]"
tune --image-path dog.webp
./run_dynamo.sh
```

This script starts the frontend and backend services, waits for them to be ready, then runs a test client to send a sample request. Once the test completes, all services are automatically shut down. This is meant as a functional check, not to provide a permanent server.

#### Dynamic batching

The service uses dynamic batching — requests are grouped and processed together for efficiency. Currently, there is one frontend and one worker. To support multiple workers, move batching to a separate service that handles request grouping.

## Model Details

Can be found in following pages:
* https://pytorch.org/vision/stable/models.html#classification
* https://huggingface.co/timm
