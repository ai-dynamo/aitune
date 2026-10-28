---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "BERT Triton Example"
---

# BERT: Torch or ONNX to Triton

Tune BERT from a Torch model or an ONNX export, publish it to Triton, and use Model Analyzer to select a
high-throughput deployment configuration. The example checks the published model through Triton gRPC.
It defaults to `bert-base-uncased` and requires Docker and an NVIDIA GPU.

```bash
cd examples/BERT
./prepare_triton.sh torch
# Or, after moving aside the previous checkpoint and model repository:
# ./prepare_triton.sh onnx
```

To use another compatible BERT model, set its Hugging Face identifier:

```bash
BERT_MODEL_NAME=google-bert/bert-base-cased ./prepare_triton.sh torch
```

The model must expose one `input_ids` input and two outputs. The example tunes batches 1, 2, and 4 at sequence
lengths 64, 128, and 256. TensorRT uses one profile covering the recorded range. Model Analyzer uses the
minimum input shape (64 tokens) for its throughput, p95, and p99 latency report; it does not aggregate the three lengths.
The report also identifies the request batch size, concurrency, and instance group used for that measurement.
Random token IDs verify numerical agreement, not language-task accuracy.

The flow saves `checkpoints/bert.ait` and publishes a Triton model named `bert` under
`model_repository/`. Each BERT command accepts `--tuned-model-path` to use another checkpoint path. Move aside
previous outputs when switching model or source. The publication repository is an output artifact, not a live
deployment target.
