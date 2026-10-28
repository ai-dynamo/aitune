---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Tune and serve WAN 2.1 on one or multiple GPUs"
---

This example tunes the `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` text-to-video pipeline with NVIDIA AITune. It demonstrates
AITune inside a multi-process Diffusers application using context parallelism. Every GPU processes a slice of the
video sequence while AITune compares compilation backends for the WAN transformer.

## What this example demonstrates

- Run WAN 2.1 with Diffusers Ulysses or ring context parallelism.
- Compare AOT and JIT compilation backends for the WAN transformer.
- Coordinate AITune inspection, correctness checks, profiling, and backend selection across all ranks.
- Save one rank-local checkpoint per process and load those checkpoints for distributed inference.
- Compare original and tuned generation time and save both results as MP4 videos.
- Serve the tuned context-parallel pipeline through one NVIDIA Dynamo endpoint.

## Environment setup

WAN 2.1 T2V 1.3B leaves enough memory on a 32 GB RTX PRO 4500 for representative inputs and backend compilation.
Context parallelism further reduces activation memory and distributes attention computation, but it does not shard
model weights; each rank loads the complete pipeline.

The recommended environment is the NVIDIA PyTorch container used by the example tests. From the repository root:

```bash
docker run --gpus all --ipc=host --rm -it \
  -v "$PWD":/workspace/aitune \
  -w /workspace/aitune/examples/WAN \
  nvcr.io/nvidia/pytorch:26.06-py3
python -m pip install uv
uv sync
```

Alternatively, install the example in your own virtual environment from `examples/WAN`:

```bash
pip install --extra-index-url https://pypi.nvidia.com .
```

## Tune on one GPU

Run the example directly to tune and save a single-GPU checkpoint:

```bash
uv run --module wan.tune
```

For a shorter local smoke run:

```bash
uv run --module wan.tune --num-frames 17 --steps 2
```

## Tune on four GPUs

Launch one process per GPU. The example detects the `torchrun` world size and enables Diffusers context parallelism
automatically:

```bash
uv run torchrun --standalone --nproc-per-node=4 \
  --log-dir logs --tee 3 --local-ranks-filter 0 \
  --module wan.tune \
  --context-parallel ulysses
```

The application initializes NCCL before loading the pipeline. Diffusers applies a context-parallel plan to the WAN
transformer. AITune detects the distributed module and resolves its default compatible candidates, currently
TorchInductor AOT and JIT, using the worst-rank performance result. Each rank saves a distinct checkpoint such as
`wan2.1-t2v-1.3b.rank-0-of-4.ait`.

Tuning uses latent pipeline output. This keeps the representative denoising workload while avoiding repeated VAE
video decoding during inspection and backend profiling.

For a short smoke run, reduce the denoising work and frame count:

```bash
uv run torchrun --standalone --nproc-per-node=4 \
  --log-dir logs --tee 3 --local-ranks-filter 0 \
  --module wan.tune \
  --context-parallel ulysses --num-frames 17 --steps 2
```

Frame counts must have the form `4k + 1`, as required by the WAN VAE. The shorter command is useful for a quick local
integration check; use the default 81 frames and 30 steps for a representative generation workload and CI parity.

## Generate and compare videos

Run inference with the same world size, context-parallel mode, model, and generation shape used for tuning:

```bash
AITUNE_OUTPUT_DIR=output uv run --module wan.inference
```

For a multi-GPU checkpoint, launch inference with the same distributed configuration used for tuning:

```bash
AITUNE_OUTPUT_DIR=output uv run torchrun --standalone --nproc-per-node=4 \
  --log-dir logs --tee 3 --local-ranks-filter 0 \
  --module wan.inference \
  --context-parallel ulysses
```

All ranks participate in generation. Rank 0 logs the end-to-end timing and writes `wan_original.mp4` and
`wan_tuned.mp4` under `AITUNE_OUTPUT_DIR` (default: `output`).

## Serve with NVIDIA Dynamo

Install the optional dependencies, tune with four GPUs, and expose the same four GPUs to the service:

```bash
uv pip install ".[dynamo]"
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_dynamo.sh
```

`run_dynamo.sh` launches one backend process per visible GPU. AITune's `DynamoWorker` detects the initialized
multi-rank process group: rank 0 owns the endpoint, requests are serialized and broadcast to all ranks, and every rank
runs the same context-parallel generation. Only rank 0 encodes and returns the final MP4. The script sends one request
to `/v1/videos`, writes `output.mp4`, and then shuts down the worker group.

Use the same visible GPU count and context-parallel mode for tuning and serving because compiled artifacts are
rank-local and placement-specific. The height, width, and frame count in `config.yaml` must also match the values used
for tuning because the checkpoint is compiled for that generation shape. Set `AITUNE_EXAMPLE_CONFIG_PATH` to use a
different serving configuration. The CI variants use the same 81-frame, 30-step generation workload as this example.

## Context-parallel modes

Ulysses is the default:

```bash
--context-parallel ulysses
```

Ring attention is also available:

```bash
--context-parallel ring
```

## Options

- `--model-name`: Hugging Face model name or path (default: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`).
- `--prompt`: Text prompt used for generation.
- `--negative-prompt`: Negative text prompt.
- `--height`, `--width`: Output dimensions (defaults: `480x832`).
- `--num-frames`: Number of frames in `4k + 1` form (default: `81`).
- `--steps`: Number of denoising steps (default: `30`).
- `--guidance-scale`: Classifier-free guidance scale (default: `5.0`).
- `--max-sequence-length`: Maximum text encoder sequence length (default: `512`).
- `--fps`: Saved video frame rate (default: `16`).
- `--tuned-model-path`: Base AITune checkpoint path (default: `wan2.1-t2v-1.3b.ait`).
- `--context-parallel`: `ulysses` or `ring` (default: `ulysses`).

## Hardware metrics

Set `AITUNE_HARDWARE_METRICS=1` to collect per-rank GPU hardware metrics:

```bash
AITUNE_HARDWARE_METRICS=1 uv run torchrun --standalone --nproc-per-node=4 \
  --module wan.tune
```

For model details, see the [WAN 2.1 T2V 1.3B model page](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers).
The distributed setup follows the
[Diffusers context-parallel inference guide](https://huggingface.co/docs/diffusers/training/distributed_inference#context-parallelism).
