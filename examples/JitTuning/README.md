<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
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
