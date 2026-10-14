<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# FLUX.1-schnell

Tune FLUX.1-schnell with AITune, compare baseline and tuned images and performance, then serve it through
Dynamo and a small web app. Both recipes use BF16, batch size 1, 1024×1024 images, and four denoising steps.
The two-GPU recipe adds Ulysses context parallelism.

## Read the example

Like [FLUX.1-dev](../FLUX), the example has two main entry points:

- [tune.py](flux_schnell_recipe/tune.py): load → `ait.inspect` → `ait.wrap` → `ait.tune` → `ait.save`.
  The transformer compares Torch-TensorRT and TorchInductor; other modules use the default AOT candidates.
- [inference.py](flux_schnell_recipe/inference.py): generate baseline images → `ait.load` into the same
  pipeline → generate tuned images. Optional modes compare RGB SSIM and benchmark both variants.

[model.py](flux_schnell_recipe/model.py) contains pipeline loading and seeded generation.
[recipes/](recipes/) holds single- and multi-GPU settings; [samples/](samples/) holds prompt/seed inputs.
`config.py` reads them, `distributed.py` handles NCCL, and `metrics.py` measures compilation and GPU use.
`dynamo/` contains the backend and client; `run_dynamo.sh` starts the service; `web.py` serves the prompt UI.

## Set up

Use Docker with NVIDIA Container Toolkit, a CUDA 13.0-compatible driver, and a BF16-capable GPU.
The pipeline and compilation candidates must fit in GPU memory; restoring artifacts also uses host RAM.
Review the [model's access requirements](https://huggingface.co/black-forest-labs/FLUX.1-schnell) and
export `HF_TOKEN` if required. From the repository root:

```bash
docker build -f examples/flux-schnell/Dockerfile -t aitune-flux-schnell .
mkdir -p results/flux-schnell hf-cache
docker run --gpus '"device=0"' --ipc=host --network=host --rm -it \
  --name aitune-flux-schnell -e HF_TOKEN \
  -v "$PWD/results/flux-schnell:/workspace/flux-schnell/results" \
  -v "$PWD/hf-cache:/cache/huggingface" aitune-flux-schnell
```

Run the following commands **inside the container**, in `/workspace/flux-schnell`.
Dependencies and the default model revision are pinned. Downloads, artifacts, results, and compiler caches
persist in the host mounts. Host networking makes the services' loopback listeners accessible from the
Docker host. Rebuild the image after changing the example.

## Tune, generate, validate, benchmark

```bash
uv run --frozen tune
uv run --frozen inference
uv run --frozen inference --mode correctness
uv run --frozen inference --mode benchmark
```

Commands default to `recipes/single-gpu.yaml`. Use `--config PATH` for another recipe, `--model-id ID` for a
compatible Hub checkpoint, or `--checkpoint DIRECTORY` for a complete local Diffusers checkpoint. Keep the
same source and recipe throughout the workflow. YAML paths are relative to the recipe file.
Changing the source or workload requires retuning.

Tuning saves `results/single-gpu/artifact/tuned.ait` and a small sidecar describing its source and recipe.
It refuses to overwrite an artifact; choose a new `artifact_path` to retune. Backend selections and build
timings are in `results/single-gpu/cache/rank-0/aitune-tuning.json`. Artifacts and sidecars from the previous
example layout can still be loaded with their original recipes.

Inference saves PNGs under `results/single-gpu/infer/{original,aitune}/images/`.
Use `--variant aitune` or `--variant original` to generate only one variant. Correctness uses the validation
samples and writes PNGs, SSIM scores, and a report under `correctness/`. Benchmarking validates the measured
pipeline and writes throughput, latency, speedup, per-GPU memory/utilization, and raw timings under `benchmark/`.
Both comparison modes require both variants.

Compilation and warmup finish before timed inference; compilation during warmup or measurement rejects the
run. Timing covers prompt-to-PIL generation and GPU synchronization, excluding PNG writes. Baseline and tuned
measurements run sequentially in one process using the same pipeline. Compiler caches are reused per recipe
and rank; restoring an artifact can still require runtime compilation. These are warm inference measurements.

**Repeated commands overwrite reports and matching PNGs.** Copy results elsewhere to retain a comparison.
Local checkpoints are identified by their path; keep their contents unchanged after tuning. Load artifacts
with compatible GPU hardware and packages.

## Use two GPUs

Expose exactly two matching BF16-capable GPUs with `--gpus '"device=0,1"'` when creating the container.
Like FLUX.1-dev, launch the modules explicitly with `torchrun`:

```bash
uv run --frozen torchrun --standalone --nproc-per-node=2 --max-restarts=0 \
  --module flux_schnell_recipe.tune --config recipes/multi-gpu.yaml

uv run --frozen torchrun --standalone --nproc-per-node=2 --max-restarts=0 \
  --module flux_schnell_recipe.inference --config recipes/multi-gpu.yaml

uv run --frozen torchrun --standalone --nproc-per-node=2 --max-restarts=0 \
  --module flux_schnell_recipe.inference --config recipes/multi-gpu.yaml --mode correctness

uv run --frozen torchrun --standalone --nproc-per-node=2 --max-restarts=0 \
  --module flux_schnell_recipe.inference --config recipes/multi-gpu.yaml --mode benchmark
```

Each rank saves its artifact; a manifest records completion of both ranks. Both GPUs cooperate on every
image, and throughput counts each image once. Rank zero saves PNGs; benchmark telemetry is saved per rank.
To check NCCL before loading the model:

```bash
uv run --frozen torchrun --standalone --nproc-per-node=2 --max-restarts=0 \
  --module flux_schnell_recipe.distributed --config recipes/multi-gpu.yaml --timeout 120
```

If direct GPU peer-to-peer communication hangs, try `export NCCL_P2P_DISABLE=1` and repeat the check.
This helped on the reported two-H100 PCIe machine. Keep it set for subsequent commands, including serving,
on an affected machine; the shared-memory fallback can affect performance.

## Dynamo and the web app

After tuning and validation, start the frontend and backend:

```bash
./run_dynamo.sh
# For two GPUs:
# ./run_dynamo.sh --config recipes/multi-gpu.yaml
```

The script waits for model loading, compilation, warmup, and registration, then prints
`Ready at http://127.0.0.1:8000`. Leave it running. In another container shell:

```bash
curl -fsS http://127.0.0.1:8000/v1/models
uv run --frozen dynamo-client
uv run --frozen dynamo-client --mode correctness
uv run --frozen dynamo-client --mode benchmark
uv run --frozen web
```

Open **http://127.0.0.1:8080** on the Docker host. Enter a prompt, generate an image, and download the PNG.
The web app uses a fresh random seed and accepts one generation at a time. For a remote Docker host, forward
port 8080 with SSH. HTTP and RPC listeners use `127.0.0.1`; Dynamo response streams use the loopback interface.

The client uses the inference samples by default. Deployment correctness compares against baseline PNGs from
`inference --mode correctness`; run that command first. Deployment benchmarking runs correctness, then AIPerf
at concurrency 1, including transport and image encoding, with GPU telemetry from the same request window.
Run the client in the same environment with the shared results directory. Benchmark samples must use the
recipe's deployment seed. Outputs go under `results/<recipe>/deployment-*/<variant>/`.

To compare a baseline service, stop Dynamo and repeat with `./run_dynamo.sh --variant original`, then add
`--variant original` to the client and web commands. For two GPUs, pass `--config recipes/multi-gpu.yaml` to
the script, client, and web app. The script also accepts `--model-id` or `--checkpoint`.
The web app and client accept `--endpoint`; the default is the recipe's local port.
Press **Ctrl-C** in each shell to stop the web app and Dynamo; the script cleans up its child processes.

API clients can send `POST /v1/images/generations` with the model ID from `/v1/models`, `prompt`, `n: 1`,
`size: "1024x1024"`, and `response_format: "b64_json"`. Optional `nvext: {"seed": 123}` chooses a seed;
omitting it uses the recipe's deployment seed. The PNG is returned in `data[0].b64_json`.

## Validation notes

The provisional hard RGB SSIM threshold is **0.70**, with warnings below **0.95**. Inspect the saved images:
the bundled prompts are a smoke workload. Replace the tuning, validation, and benchmark samples with
representative prompts for your application and calibrate the threshold. Memory requirements and
quality/performance across GPU architectures have not been established.

The previous `recipe ...` commands are replaced by `tune`, `inference`, `dynamo-client`, `web`, and
`run_dynamo.sh`. Historical run directories remain untouched; new commands use the paths described above.
