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

FP16 is the default. The recipe downloads a supplied ONNX export at a pinned revision; it does not cast weights
or apply additional quantization. Select another export with `--variant`:

```bash
./prepare_triton.sh --variant int8
```

| Variant | Supplied ONNX file |
| --- | --- |
| `fp16` (default) | `model_fp16.onnx` |
| `int8` | `model_int8.onnx` |

These are upstream export names. Quantized exports can retain floating-point operations and image inputs;
the recipe reads the image dtype from the graph rather than inferring it from the variant name.
The original FP32 `model.onnx` is excluded from selection and CI.

Each variant has a separate CI job covering tuning, checkpoint restoration, Model Analyzer, and Triton inference.
Python validation checks the restored checkpoint's input dtype against the selected source graph and verifies
detection output shape and finite values. Triton validation generates one image sample in the checkpoint's input
dtype, runs the Python checkpoint, and sends the exact same image values to Triton. It checks detection output
shape and finite values, then compares the outputs using `rtol=1e-2, atol=1e-2`. Backend build or validation
failures are logged; the workflow requires at least one deployment-capable backend to succeed.

When running commands individually, pass the same `--variant` to `yolo-tune`, `yolo-python-inference`, and
`run_triton.sh`. The model-store and Triton client commands read precision from the checkpoint.
Move aside previous checkpoints, model repositories, and Model Analyzer outputs when switching variants.

The model uses fixed batch 1. Deterministic generated input checks checkpoint execution and numerical agreement
with its Triton deployment; it does not measure detection accuracy. The flow prints the selected configuration,
request batch size, concurrency, instance group, throughput, and p95 and p99 latency.

The flow saves `checkpoints/yolov10n.ait` and publishes a Triton model named `yolov10n` under
`model_repository/`. Each YOLO command accepts `--tuned-model-path` to use another checkpoint path. The publication
repository is an output artifact, not a live deployment target. Review the model and package licenses before use.
