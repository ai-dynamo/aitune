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

# Aim

This example demonstrates how to use NVIDIA AITune to do JIT tuning without any code changes required i.e. there is no additional import to the user's code.

## Environment Setup

Install dependencies:

```bash
uv sync
```

## Usage

If you would like to run a model with JIT tuning enabled set the following environment variable:

```bash
export AUTOWRAPT_BOOTSTRAP=aitune_enable_jit_tuning
```

or set it inplace

```bash
AUTOWRAPT_BOOTSTRAP=aitune_enable_jit_tuning uv run a_command
```

If the environment variable is not set, the model will run without any tuning.


### Resnet

```bash
uv run resnet
```

### Diffusion models

#### Stable Diffusion 3


```bash
uv run diffusion --model stabilityai/stable-diffusion-3-medium-diffusers
```

#### Stable Diffusion XL

```bash
uv run diffusion --model stabilityai/stable-diffusion-xl-base-1.0
```

#### Flux

```bash
uv run diffusion --model black-forest-labs/FLUX.1-dev
```
