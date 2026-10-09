<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# FLUX.1-schnell

Tune [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) with NVIDIA AITune,
check image correctness, benchmark it, and serve custom prompts through Dynamo and a web app.
This example is a draft; broader quality and hardware evaluation remain incomplete.

The [single-GPU recipe](recipes/single-gpu.yaml) uses BF16, 1024×1024 images, four inference steps,
and one image per request. Use the same config throughout the workflow below.

## 1. Build and run the container

You need Docker with NVIDIA Container Toolkit, a driver compatible with the CUDA 13.0 stack,
and one NVIDIA GPU with native BF16 support. The full model and tuning candidates must fit in GPU memory;
artifact loading also uses host RAM. Minimum memory requirements have not been established.

Review the model's access requirements on Hugging Face and create a
[Read token](https://huggingface.co/settings/tokens). From the repository root on the host:

```bash
docker build -f examples/flux-schnell/Dockerfile \
  --build-arg EXAMPLE_DIR=examples/flux-schnell -t aitune-flux-schnell .

mkdir -p results/flux-schnell hf-cache
read -rsp "Hugging Face token: " HF_TOKEN
echo
export HF_TOKEN

docker run --gpus '"device=0"' --ipc=host --rm -it \
  --name aitune-flux-schnell \
  -e HF_TOKEN \
  -p 127.0.0.1:8000:8000 \
  -p 127.0.0.1:8080:8080 \
  -v "$PWD/results/flux-schnell:/workspace/flux-schnell/results" \
  -v "$PWD/hf-cache:/cache/huggingface" \
  aitune-flux-schnell
```

Replace GPU index `0` if needed, keeping exactly one GPU visible. If `HF_TOKEN` is already exported,
skip the token prompt. The container installs locked dependencies and opens a shell in `/workspace/flux-schnell`.
Model downloads are cached on the host; artifacts, images, logs, and reports persist under
the host's `results/flux-schnell/` directory.

For experimental two-GPU tuning, recreate the container with `--gpus '"device=0,1"'` and use
[`recipes/multi-gpu.yaml`](recipes/multi-gpu.yaml) for the Python commands. Both matching BF16-capable GPUs must
be visible inside the container with working NCCL communication. This recipe launches its own two workers;
do not wrap `recipe` in `torchrun`. Two-GPU Dynamo deployment and the web app are not supported.

Before two-GPU tuning, check communication inside the container without loading the model:

```bash
NCCL_DEBUG=INFO uv run --frozen torchrun --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
  --module flux_schnell_recipe.distributed --config recipes/multi-gpu.yaml --timeout 120
```

Expect a passing report under `results/multi-gpu/preflight/<run>/`. Resolve communication failures before tuning.
Two-GPU recipe commands stream worker logs with rank and startup-stage labels, plus elapsed time every 30 seconds.

If the check hangs using direct GPU peer-to-peer communication, set this in the container shell and repeat it:

```bash
export NCCL_P2P_DISABLE=1
```

This allowed the communication check to pass and tuning to start on the reported two-H100 PCIe machine.
Keep it set for tuning, correctness, benchmarking, and inference on that machine. Repeat the export in each new
shell, or add `-e NCCL_P2P_DISABLE=1` to `docker run`. The fallback uses host shared memory and can affect performance;
use it for affected machines rather than enabling it for every multi-GPU run.

Run the following model commands **inside the container**.

## 2. Tune the model

```bash
uv run --frozen recipe tune \
  --model-id black-forest-labs/FLUX.1-schnell --config recipes/single-gpu.yaml
```

This tunes the pipeline using the bundled inputs and saves the artifact to
`results/single-gpu/artifact/tuned.ait`, with metadata alongside it. The resolved configuration and
tuning report are under `results/single-gpu/tune/<run>/`.

Tuning refuses to overwrite an existing artifact directory. To tune again, copy the recipe and choose a new
`artifact_path` parent directory. Pass that config to every subsequent command. Paths in YAML resolve relative
to the config file. Changes to the model, image dimensions, or inference settings require retuning.

## 3. Test and benchmark

Check the saved artifact in a fresh model, then benchmark both the Original Model and AITune:

```bash
uv run --frozen recipe correctness \
  --model-id black-forest-labs/FLUX.1-schnell --config recipes/single-gpu.yaml

uv run --frozen recipe benchmark \
  --model-id black-forest-labs/FLUX.1-schnell --config recipes/single-gpu.yaml
```

Correctness compares generated images using SSIM: the current provisional threshold is `0.70`, with warnings
below `0.95`. Inspect the saved images as well; bundled prompts provide a smoke check, not a full quality evaluation.
Images and scores are under `results/single-gpu/correctness/<run>/`.

Benchmarking repeats correctness, then measures each variant in a separate process after compilation and warmup.
The report at `results/single-gpu/benchmark/<run>/report.json` includes throughput, latency, GPU memory and
utilization, compilation time, and warmup time. Workers stream progress to the terminal and retain logs in each run.

To generate images from the prompts in `samples/inference.jsonl` without starting a service:

```bash
uv run --frozen recipe infer \
  --model-id black-forest-labs/FLUX.1-schnell --config recipes/single-gpu.yaml
```

PNGs are saved to `results/single-gpu/infer/<run>/aitune/images/`.

## 4. Deploy on Dynamo

Start the service after tuning and checking correctness:

```bash
uv run --frozen recipe serve \
  --model-id black-forest-labs/FLUX.1-schnell --config recipes/single-gpu.yaml \
  --target dynamo --variant aitune
```

The service loads the artifact, completes first-use compilation and warmup, then prints
`Ready at http://localhost:8000`. It keeps the model loaded for incoming prompts. Leave this shell running.
Startup logs are in `results/single-gpu/serve-aitune/<run>/frontend.log` and `worker.log`.

From a **second host terminal**, open another shell in the same container:

```bash
docker exec -it aitune-flux-schnell bash
cd /workspace/flux-schnell
curl -fsS http://localhost:8000/v1/models
```

To check the running service and benchmark it with AIPerf, run in this second shell:

```bash
uv run --frozen recipe deployment-correctness --config recipes/single-gpu.yaml \
  --target dynamo --variant aitune --endpoint http://localhost:8000

uv run --frozen recipe deployment-benchmark --config recipes/single-gpu.yaml \
  --target dynamo --variant aitune --endpoint http://localhost:8000
```

Reports are saved under `results/single-gpu/deployment-correctness-aitune/<run>/` and
`results/single-gpu/deployment-benchmark-aitune/<run>/`. Deployment benchmarking also runs correctness first.
To compare against the Original Model, stop the service and repeat the serving and deployment checks with
`--variant original`.

## 5. Generate images in the web app

With the AITune service still running, start the web app in the second container shell:

```bash
uv run --frozen recipe web --config recipes/single-gpu.yaml --host 0.0.0.0
```

Open **http://localhost:8080** on the Docker host. Enter a prompt, select **Generate image**, and use
**Download PNG** to save the result. The page shows elapsed time and displays the generated image.
It reuses the running model, accepts one generation at a time, and uses the recipe's dimensions, steps, and seed.
Images are not saved to the results directory automatically.

The web app connects to port `8000` from the recipe by default. Use `--variant original` for an Original Model
service or `--endpoint http://HOST:PORT` for another service address. For a remote Docker host, run
`ssh -L 8080:localhost:8080 user@host` locally, then open the same browser URL. The web app is a local demo
without authentication; the Docker command above publishes its port on host loopback only.

For direct API clients, Dynamo accepts `POST /v1/images/generations` with the model ID from `/v1/models`,
a `prompt`, `n: 1`, `size: "1024x1024"`, and `response_format: "b64_json"`. The PNG is returned in `data[0].b64_json`.

Press **Ctrl-C** in the web shell to stop the app, and in the serving shell to stop Dynamo and release the model.
Rebuild the image after source changes; recreate the container if it lacks the port mappings shown above.
