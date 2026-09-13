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

## Tune ResNet

The `tune` command always creates an AITune package. Its target controls which backends are considered:

- `python` selects backends intended for local Python inference or Dynamo.
- `triton` selects TensorRT, ONNX Runtime, and TorchInductor AOT because Triton can load their artifacts directly.

Tune for Python inference and Dynamo by default:

```bash
uv run tune --target python --model-name resnet50
```

Use the Triton target when the package will be converted into a Triton model repository:

```bash
uv run tune --target triton --model-name resnet50
```

Both commands write `resnet50.ait`. Target-specific backend selection belongs in AITune itself in the future; the
example makes that choice explicitly until such an API exists.

### User-provided dynamic shapes

The dynamic-shapes mode uses the same tuning flow while explicitly defining its batch and spatial ranges. With the
defaults, it records batch sizes 1–4 at `224×224` and then runs inference at the unseen shape `(2, 3, 256, 256)`:

```bash
uv run tune --target python --dynamic-shapes 1
uv run python-inference --dynamic-shapes 1
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

To log hardware metrics during tuning or inference, export the `AITUNE_HARDWARE_METRICS=True` environment variable:

```bash
AITUNE_HARDWARE_METRICS=True uv run tune
```

## Python inference

Run the tuned package directly through AITune's Python runtime:

```bash
uv run python-inference --model-name resnet50 --image-path your_image
```

## Dynamo inference

Run ResNet as an AI Dynamo service with dynamic batching:

```bash
pip install ".[dynamo]"
tune --target python --image-path dog.webp
./run_dynamo.sh
```

This script starts the frontend and backend services, waits for them to be ready, then runs a test client to send a sample request. Once the test completes, all services are automatically shut down. This is meant as a functional check, not to provide a permanent server.

With the services already running, invoke the Dynamo client independently:

```bash
uv run --extra dynamo dynamo-inference --num-requests 1
```

### Dynamic batching

The service uses dynamic batching — requests are grouped and processed together for efficiency. Currently, there is one frontend and one worker. To support multiple workers, move batching to a separate service that handles request grouping.

## Triton inference

For local testing, use the same NVIDIA monthly release for tuning and serving. The preparation script runs tuning
inside the PyTorch container and generates the model repository from the resulting `resnet50.ait` package:

```bash
export NVIDIA_RELEASE=26.05
./prepare_triton.sh
```

It uses `nvcr.io/nvidia/pytorch:${NVIDIA_RELEASE}-py3`. Start the matching
`nvcr.io/nvidia/tritonserver:${NVIDIA_RELEASE}-py3` server and run the test client with:

```bash
uv sync --extra triton
uv run --extra triton ./run_triton.sh
```

The runner waits for Triton, invokes the client, and stops the server. Both scripts default to release `26.05`; set
`NVIDIA_RELEASE` once in the calling shell to use another matching pair.

Triton loads the published models at startup with `--model-control-mode=none`; no load API call is needed.
The runner uses host networking locally and shares the job container's network in GitLab Docker jobs, so the
readiness checks and client reach the same `localhost`. When running inside another Docker container, set
`TRITON_NETWORK=container:<name-or-id>` to share that container's network.

The equivalent package-to-repository command is available independently:

```bash
uv run --extra triton triton-model-store --tuned-model-path resnet50.ait
```

Model-store generation loads the package, extracts its selected artifact, and creates
`model_repository/resnet50`. The `config.pbtxt`, tensor bounds, and maximum batch size all come from that artifact.
It also creates `model_analyzer/fast.yaml` for a quick search and `model_analyzer/manual.yaml` for the complete
recommended search space. Neither operation replaces an existing output directory.

AITune backends that require a Python process are intentionally excluded from this flow. Deploy those through the
Dynamo path above instead of wrapping them in Triton's Python Backend.

With Triton already running, invoke the client independently:

```bash
uv run --extra triton triton-inference --model-name resnet50
```

The client obtains the exported input and output names from server metadata.

Keeping releases aligned is required for TensorRT plans: by default, a plan only loads with the exact TensorRT
version that built it and the same GPU compute capability. Matching releases also keeps the PyTorch runtime aligned
for AOTInductor artifacts.

To optimize the deployment, install Triton Model Analyzer and start with the generated fast configuration:

```bash
model-analyzer profile --config-file model_analyzer/fast.yaml
```

Use `manual.yaml` afterward when a wider search over batch sizes, dynamic batching, queue delay, and model instance
count is worth the additional profiling time.

Compatibility references:

- [TensorRT engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)
- [Triton Inference Server release notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html)
- [NVIDIA PyTorch container release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/)

## Model Details

Can be found in following pages:
* https://pytorch.org/vision/stable/models.html#classification
* https://huggingface.co/timm
