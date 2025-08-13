<!--
Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Nemo ASR Parakeet RNNT 1.1B Pipeline Tuning with NVIDIA AITune

This example demonstrates how to use NVIDIA AITune to tune the Nemo ASR with Parakeet RNNT 1.1B model.

## Environment Setup

You can use either of the following options to setup the environment:

### Option 1 - virtual environment managed by you

Activate your virtual environment and install the dependencies:

```bash
pip install .
```

### Option 2 - virtual environment managed by `uv`

Install dependencies:

```bash
uv sync
```

## Usage

### Tuning and inference the model

To tune the ASR model, run:

```bash
tune --audio_path 2086-149220-0033.wav
```

To infer the ASR model, run:

```bash
inference --audio_path 2086-149220-0033.wav
```


## Model Details

Can be found in following pages:
* https://huggingface.co/nvidia/parakeet-rnnt-1.1b
* https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/asr/models.html