---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Tune and serve FLUX.1-schnell using recipes"
---

This example inspects, tunes, validates, and serves the FLUX.1-schnell Diffusers pipeline. YAML files in `recipes/` are
its configuration source. Commands accept `--recipe`; model, workload, input, output, GPU, and serving settings are
read from that file. With no argument, commands select `recipes/single-gpu.yaml` from the example working directory.

## Recipes and samples

- `recipes/single-gpu.yaml` uses one GPU and is the recipe for this machine.
- `recipes/multi-gpu.yaml` uses two GPUs on one node. It is prepared for context-parallel execution; it has not been
  validated on this single-GPU machine.

All paths in a recipe are resolved relative to the YAML file, including `artifact_path`, `output_dir`, and `inputs`.
The model is loaded at the exact `model.revision`. Each JSONL sample has an `id`, a `prompt`, and a `seed`:

| Recipe input | Purpose |
| --- | --- |
| `inputs.tune` | Every sample is used for inspection and tuning. |
| `inputs.inference` | Original/tuned inference images and Dynamo client requests. |
| `inputs.validation` | Held-out original/tuned images used for correctness validation. |
| `inputs.benchmark` | Samples cycled through during warmup and timed measurements. |

The supplied files contain two tuning prompts, four validation prompts, and one inference prompt. The generator is
reset to the sample's seed on every call, so warmup does not change the measured image.

Generation settings come from `workload`. `execution.gpu_count` controls the number of worker processes, and
`execution.context_parallel` selects `ulysses` or `ring`. `tuning.quantization` optionally includes TorchAO NVFP4 and
FP8 transformer candidates; the supplied recipes leave it disabled. NVFP4 needs compatible Blackwell hardware.

## Build and run in a container

From the repository root:

```bash
docker build -f examples/FluxSchnell/Dockerfile -t aitune-flux-schnell:recipes .
mkdir -p examples/FluxSchnell/results

docker run --rm --gpus all --ipc=host \
  -e HF_TOKEN -e HF_HOME=/cache/huggingface \
  -v "$PWD/hf-cache":/cache/huggingface \
  -v "$PWD/examples/FluxSchnell/results":/workspace/aitune/examples/FluxSchnell/results \
  aitune-flux-schnell:recipes tune --recipe recipes/single-gpu.yaml

docker run --rm --gpus all --ipc=host \
  -e HF_TOKEN -e HF_HOME=/cache/huggingface \
  -v "$PWD/hf-cache":/cache/huggingface \
  -v "$PWD/examples/FluxSchnell/results":/workspace/aitune/examples/FluxSchnell/results \
  aitune-flux-schnell:recipes inference --recipe recipes/single-gpu.yaml
```

The Dockerfile accepts `--build-arg BASE_IMAGE=...` to reuse a compatible CUDA image with dependencies installed.
For a local environment, run `uv sync --extra dynamo` from `examples/FluxSchnell`, then use:

```bash
uv run tune --recipe recipes/single-gpu.yaml
uv run inference --recipe recipes/single-gpu.yaml
```

Tuning compares Torch-TensorRT JIT and TorchInductor JIT for the transformer, plus optional TorchAO candidates.
Other discovered modules use AITune's default backend strategy. Maximum-batch-size discovery is disabled to match
the single-image workload. Each rank saves its own checkpoint; rank zero writes `tuned.manifest.json` only after all
ranks have saved. Loading checks the model revision, workload, execution settings, and tuning settings against that
manifest. Checkpoint tensors are staged on CPU to avoid duplicating the weights on the GPU during loading.

## Inference, validation, and timing

`inference` generates original and tuned images from `inputs.inference`, validates using `inputs.validation`, and
benchmarks using `inputs.benchmark`. Results are saved under the recipe's `output_dir`:

- `inference/{original,tuned}/<sample-id>.png`
- `validation/{original,tuned}/<sample-id>.png`
- `report.json`, with sample IDs, seeds, RGB SSIM scores, latency distributions, GPU memory samples, and speedup.

RGB SSIM averages structural similarity over the three RGB channels using a 7×7 window and pixel values in `[0, 1]`.
`validation.threshold` determines pass/fail; scores below `validation.soft_threshold` produce warnings. A failed hard
threshold makes the command exit unsuccessfully after writing the report. Maximum absolute and relative pixel errors
are also reported, but do not replace the recipe's SSIM acceptance criterion.

`benchmark.warmup_iterations` and `benchmark.repetitions` are total iteration counts per original/tuned pipeline,
cycling through the benchmark samples in file order. Timings synchronize all participating GPUs. Memory is sampled
per rank at `benchmark.sampling_interval_s`; peak allocated GPU memory is also recorded.

## Dynamo and the prompt UI

After tuning, start the persistent deployment from the repository root:

```bash
docker run -d --name aitune-flux-schnell-web --gpus all --ipc=host \
  -p 127.0.0.1:8080:8080 \
  -e HF_TOKEN -e HF_HOME=/cache/huggingface \
  -v "$PWD/hf-cache":/cache/huggingface \
  -v "$PWD/examples/FluxSchnell/results":/workspace/aitune/examples/FluxSchnell/results \
  aitune-flux-schnell:recipes bash run_dynamo.sh --recipe recipes/single-gpu.yaml --serve
```

Open [http://localhost:8080](http://localhost:8080) when the log prints `Web app:`. Dynamo uses `deployment.port`; the
UI uses `deployment.web_host` and `deployment.web_port`. If you change the web port, update the Docker port mapping.
Both services run inside the container. Model loading and compilation can take several minutes; startup is bounded
by `deployment.readiness_timeout_s`.

The UI serves the packaged `flux_schnell/dynamo/web.html`. It uses the recipe's dimensions and steps, and chooses a
random seed for each prompt. API clients can send `nvext: {"seed": 123}`; requests without a seed use
`deployment.seed`. Requests must use the recipe's image dimensions and generate one image at a time.

Without `--serve`, `run_dynamo.sh` waits for readiness, sends `deployment.request_count` requests using the inference
samples and their seeds, saves PNGs and timings under `output_dir/deployment`, then shuts the services down. To run
that client against the existing service, execute this inside the container:

```bash
python -m flux_schnell.dynamo.client --recipe recipes/single-gpu.yaml
```

Use `docker logs -f aitune-flux-schnell-web` to follow startup and `docker stop --timeout 60 aitune-flux-schnell-web` to
stop the deployment.

## Multiple GPUs

Use `--recipe recipes/multi-gpu.yaml` for tuning, inference, and deployment on a machine exposing at least two GPUs.
Each command automatically launches the configured number of ranks with `torchrun`; additional visible GPUs are not
used. There is no separate GPU-count CLI override. Explicit external `torchrun` launches are also accepted, provided
`WORLD_SIZE` equals the recipe's GPU count. The backend configures Diffusers context parallelism on every rank, and
AITune coordinates distributed tuning and serving. Only rank zero writes images and reports.

Use the same recipe and GPU count for tuning and serving. The multi-GPU recipe saves separate artifacts under
`results/multi-gpu` and never shares the single-GPU checkpoint.
