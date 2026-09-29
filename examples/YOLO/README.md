---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "YOLO Triton Example"
---

# YOLOv10n: ONNX to Triton

Tune the pinned [`onnx-community/yolov10n`](https://huggingface.co/onnx-community/yolov10n) ONNX model,
publish it to Triton, and use Model Analyzer to select a high-throughput deployment configuration.
The example checks the published model through Triton gRPC. It requires Docker and an NVIDIA GPU.

```bash
cd examples/YOLO
./prepare_triton.sh
```

The model uses fixed batch 1. Deterministic generated input verifies numerical agreement between the source,
tuned checkpoint, and Triton deployment; it does not measure detection accuracy. The flow prints the selected
configuration, request batch size, concurrency, instance group, throughput, and p95 and p99 latency.

The flow saves `checkpoints/yolov10n.ait` and publishes a Triton model named `yolov10n` under
`model_repository/`. Each YOLO command accepts `--tuned-model-path` to use another checkpoint path. The publication
repository is an output artifact, not a live deployment target. Review the model and package licenses before use.
