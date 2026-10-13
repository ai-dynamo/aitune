<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# FLUX.1-schnell

Tune the FLUX.1-schnell Diffusers pipeline with NVIDIA AITune, compare its images and performance,
and serve it through Dynamo with a small web app. Both recipes use BF16, batch size 1, 1024×1024
images, and four denoising steps. The two-GPU recipe adds Ulysses context parallelism.

## Start here in the code

Read [tune.py](flux_schnell_recipe/tune.py) first: it loads the pipeline, calls `ait.inspect`, wraps
the transformer and remaining modules with their tuning strategies, calls `ait.tune`, then `ait.save`.
The transformer compares Torch-TensorRT and TorchInductor; other modules use AITune's default AOT candidates.
[model.py](flux_schnell_recipe/model.py) shows loading the saved artifact into a fresh pipeline and seeded generation.

| Files | Purpose |
|---|---|
| `recipes/`, `samples/`, `config.py` | Workload settings and prompt/seed inputs |
| `tune.py`, `model.py` | The AITune workflow |
| `evaluate.py`, `metrics.py` | Image validation, latency/throughput, compilation timing, GPU sampling |
| `runtime.py`, `worker.py`, `distributed.py` | One execution path for single- and multi-GPU jobs |
| `dynamo/`, `web.py`, `web.html` | Serving, deployment checks, and the prompt UI |
| `records.py`, `cli.py` | Artifact metadata, run directories, and commands |

## Set up

Use Docker with NVIDIA Container Toolkit, a CUDA 13.0-compatible driver, and a BF16-capable GPU.
The model and compilation candidates must fit in GPU memory; artifact loading also uses host RAM.
Minimum memory requirements have not been established. Review the
[model's access requirements](https://huggingface.co/black-forest-labs/FLUX.1-schnell) and export
`HF_TOKEN` if required. From the repository root:

```bash
docker build -f examples/flux-schnell/Dockerfile -t aitune-flux-schnell .
mkdir -p results/flux-schnell hf-cache
docker run --gpus '"device=0"' --ipc=host --rm -it \
  --name aitune-flux-schnell -e HF_TOKEN \
  -p 127.0.0.1:8000:8000 -p 127.0.0.1:8080:8080 \
  -v "$PWD/results/flux-schnell:/workspace/flux-schnell/results" \
  -v "$PWD/hf-cache:/cache/huggingface" aitune-flux-schnell
```

The following commands run **inside the container**, in `/workspace/flux-schnell`. Dependencies are
locked; the model cache and results persist on the host. Rebuild the image after changing the example.

## Tune, validate, benchmark, generate

```bash
uv run --frozen recipe tune
uv run --frozen recipe correctness
uv run --frozen recipe benchmark
uv run --frozen recipe infer
```

Commands default to [recipes/single-gpu.yaml](recipes/single-gpu.yaml) and its pinned model revision.
Use `--config PATH` to select another recipe, `--model-id ID` for another compatible Hub checkpoint,
or `--checkpoint DIRECTORY` for a complete local Diffusers checkpoint. Use the same source and recipe
throughout tuning, validation, inference, and serving.

Tuning saves `results/single-gpu/artifact/tuned.ait` and its metadata. It refuses to overwrite an existing
artifact directory: copy the recipe and choose a new `artifact_path` parent to retune. YAML paths are relative
to the recipe file. Changing the model, image dimensions, or inference settings requires retuning.
Backend selections and build timings are in AITune's `aitune-tuning.json` within the tuning run.

Correctness generates reference and tuned PNGs in fresh processes and compares RGB SSIM. Benchmarking runs
correctness first, then measures each variant in a fresh process after first-use compilation and warmup.
Reports include throughput, latency, GPU memory/utilization, and compilation/warmup time; compilation during
measurement rejects the run. Inference uses the prompts in `samples/inference.jsonl`.

Each command prints its output path under `results/<recipe>/<command>/<run>/`. Worker logs stream to the
terminal and remain with the reports. Single-GPU inference images are in `infer/<run>/aitune/images/`.

## Use two GPUs

Recreate the container with `--gpus '"device=0,1"'` and pass `--config recipes/multi-gpu.yaml` to every command:

```bash
uv run --frozen recipe tune --config recipes/multi-gpu.yaml
uv run --frozen recipe correctness --config recipes/multi-gpu.yaml
uv run --frozen recipe benchmark --config recipes/multi-gpu.yaml
uv run --frozen recipe infer --config recipes/multi-gpu.yaml
```

Expose exactly two matching BF16-capable GPUs. The CLI launches both ranks; do not wrap `recipe` in `torchrun`.
Each GPU saves its own artifact, and a manifest records the completed tuning run. Existing artifacts remain
compatible. Both GPUs cooperate on each image; throughput counts each image once. Per-rank reports and images
are under `rank-0/` and `rank-1/` within each variant directory.

To check NCCL communication before loading the model:

```bash
uv run --frozen torchrun --standalone --nproc-per-node=2 --max-restarts=0 \
  --module flux_schnell_recipe.distributed --config recipes/multi-gpu.yaml --timeout 120
```

If direct GPU peer-to-peer communication hangs, try `export NCCL_P2P_DISABLE=1` and repeat the check.
This helped on the reported two-H100 PCIe machine. On an affected machine, keep the variable set for all
subsequent commands, including serving. The shared-memory fallback can affect performance.

## Serve through Dynamo

After tuning and checking correctness:

```bash
uv run --frozen recipe serve
```

The service loads the artifact and finishes compilation and warmup before printing `Ready at http://localhost:8000`.
Leave it running. Open a second shell with `docker exec -it aitune-flux-schnell bash`, then run:

```bash
uv run --frozen recipe deployment-correctness
uv run --frozen recipe deployment-benchmark
uv run --frozen recipe web --host 0.0.0.0
```

Open **http://localhost:8080** on the Docker host, enter a prompt, and select **Generate image**.
Each generation uses a fresh random seed. The page displays elapsed time and a **Download PNG** button;
it accepts one generation at a time and does not automatically save images to the results directory.

Deployment benchmarking runs correctness first and uses AIPerf at concurrency 1. To compare against the
Original Model, stop the service and repeat serving and deployment checks with `--variant original`.
For two GPUs, add `--config recipes/multi-gpu.yaml` to serving, deployment checks, and the web command.
Dynamo exposes one endpoint; both ranks participate in each request and must finish warmup before readiness.

The endpoint defaults to the recipe's port; override it with `--endpoint http://HOST:PORT`. Deployment measurements
require a local service and shared results directory; the web app can connect remotely. For a remote Docker host,
forward port 8080 with SSH. This is a local demo without authentication; the Docker command binds ports to host loopback.

API clients can use `POST /v1/images/generations` with the model ID from `/v1/models`, a `prompt`, `n: 1`,
`size: "1024x1024"`, and `response_format: "b64_json"`. Optional `nvext: {"seed": 123}` chooses a seed;
omitting it uses the recipe's deployment seed. The PNG is returned in `data[0].b64_json`.
Press **Ctrl-C** in each shell to stop the web app and service; service shutdown also stops all workers.

## Evaluation notes

The provisional hard SSIM threshold is **0.70**, with warnings below **0.95**. Inspect the saved images as well:
the bundled prompts are a smoke workload, not a representative quality evaluation. Replace the separate tuning,
validation, and benchmark inputs with representative prompts for your application and calibrate the threshold.

[Historical evaluation records](results/README.md) cover one RTX 6000 Ada. Broader quality/performance evaluation,
another GPU architecture, and the complete two-H100 serving workflow remain unvalidated. Retain resolved recipes,
environment/build information, artifacts, PNGs, and raw measurements with new results. GPU architecture and package
versions must match the artifact's build environment. Runtime compilation timing covers compiler-frame intervals
(including tracing and lowering), separately from tuning/search, artifact loading, and first-use duration.
