---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "BERT Triton Example"
---

# BERT: Torch or ONNX to Triton

This example follows the [ResNet Triton workflow](../ResNet/README.md#triton-inference): tune a source model, save
an AITune checkpoint, publish a Triton model repository, profile it with Model Analyzer, promote the
highest-throughput configuration, and check inference through Triton gRPC. It uses BERT-base-uncased and extends
the input contract from [`006_onnx_bert.py`](../../tests/functional/onnx/006_onnx_bert.py) to 64, 128, and 256
tokens.

Choose an origin on a CUDA-capable host with Docker and an NVIDIA GPU:

```bash
cd examples/BERT
./prepare_triton.sh torch
# Or, in a fresh artifacts directory:
# ./prepare_triton.sh onnx
```

The ONNX origin exports the pretrained Torch model if no ONNX file is supplied. To start from an existing graph, set
`ONNX_PATH` to a file visible inside the container (the repository is mounted at `/workspace`):

```bash
ONNX_PATH=/workspace/path/to/bert-base-uncased.onnx ./prepare_triton.sh onnx
```

An existing ONNX graph must support dynamic batch and sequence axes for all configured shapes.

The same flow can be run step by step inside a compatible Triton container after `pip install -e '.[triton]'` and
`./install.sh`:

```bash
bert-tune --source torch
bert-python-inference
bert-triton-model-store
./run_triton.sh
```

Use `--source onnx` for the ONNX path. Tuning records batches 1, 2, and 4 at each of the three sequence lengths.
TensorRT receives a profile for each recorded shape and a fallback covering batches 1–4 and sequence lengths
64–256. Tuning checks the selected backend against the source model. The separate Python command checks the saved
checkpoint, and the Triton client compares deployment outputs with that checkpoint, including batch 3 to check the
fallback. Synthetic token IDs check numerical correctness, not language-task accuracy.

Model Analyzer receives one input-data file with 64-, 128-, and 256-token requests and selects a configuration for
that mixed workload. Its throughput and p99 latency are aggregate measurements, not separate results for each
sequence length. The selected config and measurements are printed by the example-local promotion script.

Both origins write `artifacts/bert.ait` and publish a Triton model named `bert`. Publication and promotion do not
replace existing models or deployment repositories; use a fresh artifacts directory when changing origin or rerunning
the flow. Do not use the publication repository as a live Triton deployment target. Functional CI runs the Torch and
ONNX origins as separate Triton jobs.
