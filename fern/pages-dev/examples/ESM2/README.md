---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "ESM2 Model Tuning"
---

> Evolutionary-scale prediction of atomic level protein structure with a language model

This example demonstrates how to use NVIDIA AITune to tune the ESM2 transformer protein language model - `facebook/esm2_t33_650M_UR50D` - from Hugging Face's transformer library.

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

### Tuning the model

To tune the ESM2 model, run:

```bash
tune
```

The example saves the tuned checkpoint under `checkpoints/esm2_tuned.ait` and then copies the archive plus SHA sidecar to `/tmp/esm2_tuned.ait`.

After tuning, run inference

```bash
inference
```

`inference` loads the relocated checkpoint from `/tmp/esm2_tuned.ait`.

### AI Dynamo ESM2 Deployment

To run ESM2 as AI Dynamo service, use the helper script:

```sh
uv pip install ".[dynamo]"
tune
./run_dynamo.sh
```

#### Dynamic batching

The service uses dynamic batching — requests are grouped and processed together for efficiency. Currently, there is one frontend and one worker. To support multiple workers, move batching to a separate service that handles request grouping.


## Model Details

ESM-2 (Evolutionary Scale Modeling-2) is a state-of-the-art protein language model developed by Facebook AI, designed to analyze and interpret protein sequences using deep learning techniques. It is trained on a masked language modeling objective, meaning it predicts missing amino acids in protein sequences, which enables it to learn patterns relevant for understanding structure and function.

## Links

* [Hugging Face Model](https://huggingface.co/facebook/esm2_t33_650M_UR50D)
* [Research Paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2)
