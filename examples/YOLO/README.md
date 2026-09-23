---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "YOLO Triton Example"
---

# YOLOv10n: Torch or ONNX to Triton

This example follows the [ResNet Triton workflow](../ResNet/README.md#triton-inference): tune a source model, save
an AITune checkpoint, publish a Triton model repository, profile it with Model Analyzer, promote the
highest-throughput configuration, and check detections through Triton gRPC. The ONNX origin uses the pinned
[`onnx-community/yolov10n`](https://huggingface.co/onnx-community/yolov10n) graph and street image from
[`007_onnx_yolo.py`](../../tests/functional/onnx/007_onnx_yolo.py). The Torch origin uses Ultralytics
`yolov10n.pt`; these weights are a separate download and are checked against their own source output.

Choose an origin on a CUDA-capable host with Docker and an NVIDIA GPU:

```bash
cd examples/YOLO
./prepare_triton.sh torch
# Or, in a fresh artifacts directory:
# ./prepare_triton.sh onnx
```

For a custom graph or Torch weights, set `ONNX_PATH` or `TORCH_WEIGHTS` to a file visible inside the container. The
repository is mounted at `/workspace`:

```bash
ONNX_PATH=/workspace/path/to/model.onnx ./prepare_triton.sh onnx
TORCH_WEIGHTS=/workspace/path/to/yolov10n.pt ./prepare_triton.sh torch
```

The same flow can be run step by step inside a compatible Triton container after `pip install -e '.[triton]'` and
`./install.sh`:

```bash
yolo-tune --source onnx
yolo-python-inference
yolo-triton-model-store
./run_triton.sh
```

Use `--source torch` for the Torch path. Both origins use fixed batch 1 and no dynamic batching. Tuning checks the
selected backend against the source model. The separate Python command checks the saved checkpoint, and the Triton
client compares deployment detections above confidence 0.4 with that checkpoint.
The example-local promotion script prints the selected Triton config, throughput, and p99 latency. Review the model
and package licenses before use.

Both origins write `artifacts/yolov10n.ait` and publish a Triton model named `yolov10n`. Publication and promotion
do not replace existing models or deployment repositories; use a fresh artifacts directory when changing origin or
rerunning the flow. Do not use the publication repository as a live Triton deployment target. Functional CI runs the
Torch and ONNX origins as separate Triton jobs.
