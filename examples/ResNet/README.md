---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "ResNet models tuning"
---

This example demonstrates how to use NVIDIA AITune to tune a ResNet model.
ResNet-50 uses FP16 weights and inputs on CUDA for tuning and inference in the Python, Dynamo, and Triton workflows.

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

For local testing, the preparation script runs the complete workflow in a full Triton Server container. It installs
the matching PyTorch tuning stack, tunes the model, generates the model repository, profiles its configurations with
Model Analyzer, promotes the configuration with the highest measured throughput, starts Triton from the deployment
repository, and validates an inference request:

```bash
export NVIDIA_RELEASE=26.05
./prepare_triton.sh
```

For Triton, AITune searches for the maximum batch size, then chooses the highest-throughput backend and batch size
whose mean inference latency is at most 50 ms. The same numeric budget is passed to Model Analyzer as a p99 service
latency constraint; promotion rejects configurations that exceed it. Mean model latency and p99 service latency are
different measurements, so neither search guarantees production latency without testing on the deployment hardware.

It uses `nvcr.io/nvidia/tritonserver:${NVIDIA_RELEASE}-py3` and defaults to release `26.05`. This image supplies the
CUDA 13.2.1, TensorRT 10.16.1.11, and ONNX Runtime 1.24.4 libraries used for both tuning and serving. The example
installs the matching PyTorch 2.12 tuning stack without replacing those runtime libraries. Model Analyzer runs Triton
and Perf Analyzer from `/opt/tritonserver/bin`. The validation script then starts
`/opt/tritonserver/bin/tritonserver` in the current container, waits for the promoted model to become ready, invokes
the client, and stops the server on exit. Set `TRITONSERVER` when the executable is installed elsewhere.

Triton loads the published models at startup with `--model-control-mode=none`; no load API call is needed.
The server and client run sequentially in the same CI job container and communicate over `localhost`.
CI runs two Triton jobs: one uses the prebuilt image without changing its runtime packages, and the other uses a
runner-compatible raw Triton image. In the raw-image job, the functional executor installs the example dependencies,
runs `install.sh` to select the container's TensorRT and CUDA 13 ONNX Runtime packages, and then runs the same tuning,
Model Analyzer, and inference checks. This also validates the installer used by `prepare_triton.sh` locally.

The equivalent package-to-repository command is available independently:

```bash
uv run --extra triton triton-model-store --tuned-model-path resnet50.ait
```

Model-store generation loads the package, extracts its selected artifact, and creates
`model_repository/resnet50`. The `config.pbtxt`, tensor bounds, and maximum batch size all come from that artifact.
It also creates `model_repository/resnet50/model_analyzer/config.yaml` for a bounded quick search. Publication does
not replace an existing model directory. Generation writes directly into that directory; if it fails, remove or
archive the incomplete directory before retrying. The automated Triton workflow invokes `run_triton.sh`, so profiling
must complete successfully before the inference validation starts.

The command explicitly deactivates the loaded module before exiting to release its backend runtime.

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

The preparation and automated test workflows run the equivalent profiling and promotion commands:

```bash
model-analyzer profile \
  --config-file model_repository/resnet50/model_analyzer/config.yaml \
  --triton-server-path /opt/tritonserver/bin/tritonserver \
  --perf-analyzer-path /opt/tritonserver/bin/perf_analyzer
uv run --extra triton triton-promote \
  --analyzer-config model_repository/resnet50/model_analyzer/config.yaml \
  --deployment-model-repository model_repository-deployment
```

Model Analyzer writes its measurements and generated configuration variants outside the source repository under
`model_repository-model-analyzer/resnet50`. Model Analyzer's Triton configuration search retains the 50 ms p99
latency budget. The promotion command selects the compliant ResNet configuration with the highest measured throughput,
prints the selected `config.pbtxt`, copies the original model files into `model_repository-deployment/resnet50`, and
installs that config.
Model Analyzer and promotion refuse to overwrite existing output, so remove or archive previous results before
repeating the workflow.

Compatibility references:

- [TensorRT engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)
- [Triton Inference Server release notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html)
- [NVIDIA PyTorch container release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/)

## Model Details

Can be found in following pages:
* https://pytorch.org/vision/stable/models.html#classification
* https://huggingface.co/timm
