---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Tune FLUX on one or multiple GPUs"
---

This example tunes the `black-forest-labs/FLUX.1-dev` Diffusers pipeline with NVIDIA AITune. It demonstrates the
complete ahead-of-time workflow: inspect the pipeline, tune selected modules, save the result, load it into a fresh
pipeline, and compare original and tuned image generation.

## What this example demonstrates

- Tune a FLUX pipeline on one GPU.
- Select the highest-throughput backend independently for the transformer and other tunable modules.
- Save and load an AITune checkpoint.
- Compare original and tuned generation time and output images.
- Tune and run the transformer across multiple GPUs with Diffusers context parallelism.
- Optionally evaluate TorchAO NVFP4 and FP8 transformer backends on one GPU.
- Serve the tuned multi-GPU pipeline through NVIDIA Dynamo.

## Environment setup

FLUX.1-dev is a gated model. Accept its Hugging Face license and make a Hugging Face token available in the
environment before running the example.

The recommended environment is the NVIDIA PyTorch container used by the example tests. From the repository root:

```bash
docker run --gpus all --ipc=host --rm -it \
  -v "$PWD":/workspace/aitune \
  -w /workspace/aitune/examples/FLUX \
  nvcr.io/nvidia/pytorch:26.06-py3
python -m pip install uv
uv sync
```

Alternatively, install the example in your own virtual environment from `examples/FLUX`:

```bash
pip install --extra-index-url https://pypi.nvidia.com .
```

## Tune on one GPU

```bash
uv run tune \
  --model-name black-forest-labs/FLUX.1-dev \
  --prompt "A futuristic cityscape with neon lights"
```

The command:

1. Uses `aitune.torch.inspect` to find tunable pipeline modules.
2. Wraps the transformer separately from the remaining modules.
3. Profiles the configured backends and selects the highest-throughput backend for each module.
4. Saves the tuned pipeline as an AITune checkpoint.

The transformer compares Torch-TensorRT and TorchInductor. Other tunable modules compare TensorRT and
TorchInductor backends.

### Optional transformer quantization

On one GPU, add `--quantization` to include TorchAO NVFP4 and FP8 dynamic-quantization candidates for the transformer:

```bash
uv run tune --quantization
```

The other pipeline modules remain unquantized. NVFP4 requires a Blackwell GPU with CUDA capability 10.0 or newer.
This example currently exercises quantization only in its single-GPU configuration.

## Generate and compare images on one GPU

After tuning, run inference with the same model and generation settings:

```bash
AITUNE_OUTPUT_DIR=output uv run inference \
  --prompt "A beautiful landscape with mountains and a lake"
```

The inference command loads a fresh pipeline, generates an original image, loads the AITune checkpoint, and generates
a tuned image. It logs both generation times and saves both images under `AITUNE_OUTPUT_DIR` (default: `output`).

## Tune on multiple GPUs

Launch one process per GPU and add `--multi-gpu`. This tested example uses four GPUs:

```bash
uv run torchrun --standalone --nproc-per-node=4 \
  --log-dir logs --tee 3 --local-ranks-filter 0 \
  --module flux.tune \
  --multi-gpu --context-parallel ulysses
```

The application initializes NCCL and enables Diffusers context parallelism on the transformer. AITune detects that
existing distributed environment and coordinates inspection and tuning across all ranks. Each rank saves its own
checkpoint artifact so processes never write to the same file.

Run inference with the same GPU count and context-parallel mode used for tuning:

```bash
AITUNE_OUTPUT_DIR=output uv run torchrun --standalone --nproc-per-node=4 \
  --log-dir logs --tee 3 --local-ranks-filter 0 \
  --module flux.inference \
  --multi-gpu --context-parallel ulysses
```

Every rank participates in generation; rank 0 records timings and saves the images.

### Keep distributed logs readable

By default, `torchrun` sends output from every worker to the same console. The commands above use `--tee 3` to retain
stdout and stderr from every worker under `--log-dir`, while `--local-ranks-filter 0` keeps the interactive console
focused on local rank 0. On a multi-node run, local rank 0 from each node remains visible. This launcher-level approach
keeps application logging unchanged and preserves per-rank diagnostics for investigation.

## Options

- `--model-name`: Hugging Face model name or path (default: `black-forest-labs/FLUX.1-dev`).
- `--prompt`: Text prompt used for inspection, tuning, or inference.
- `--sizes`: Space-separated `width,height` pairs (default: `1024,1024`).
- `--steps`: Number of diffusion steps (default: `28`).
- `--guidance-scale`: Guidance scale (default: `3.5`).
- `--max-sequence-length`: Maximum text sequence length (default: `128`).
- `--tuned-model-path`: Base checkpoint path (default: `flux-dev.ait`).
- `--multi-gpu`: Enable Diffusers context parallelism.
- `--context-parallel`: `ulysses` or `ring` (default: `ulysses`).
- `--quantization`: Add TorchAO NVFP4 and FP8 transformer candidates.

## Serve the tuned pipeline with NVIDIA Dynamo

Install the optional dependencies, tune with four GPUs, and expose the same four GPUs to the test service:

```bash
uv pip install ".[dynamo]"
uv run torchrun --standalone --nproc-per-node=4 \
  --log-dir logs --tee 3 --local-ranks-filter 0 \
  --module flux.tune \
  --multi-gpu --context-parallel ulysses
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_dynamo.sh
```

`run_dynamo.sh` starts the Dynamo frontend and one backend rank per visible GPU, waits for the model endpoint, sends one
OpenAI-compatible image request, saves the response to `output.png`, and shuts everything down. Rank 0 owns the Dynamo
endpoint and broadcasts each request to the follower ranks so all GPUs participate.

Use the same visible GPU count for tuning and serving. The script is a functional deployment example, not a permanent
server. It processes one request at a time, matching the batch-size-1 profiles produced during tuning.

## Hardware metrics

Set `AITUNE_HARDWARE_METRICS=1` to collect GPU hardware metrics during tuning or inference:

```bash
AITUNE_HARDWARE_METRICS=1 uv run inference
```

For model details, see the [FLUX.1-dev model page](https://huggingface.co/black-forest-labs/FLUX.1-dev). The multi-GPU
setup follows the [Diffusers context-parallel inference guide](https://huggingface.co/docs/diffusers/training/distributed_inference#context-parallelism).
