<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# FLUX.1-schnell

## Model information

[FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) generates images from text using
a timestep-distilled transformer, CLIP and T5 text encoders, and a VAE. This example loads a complete
Diffusers pipeline in BF16 and tunes it with NVIDIA AITune 0.6.0.

| Item | Configuration |
|---|---|
| Task | Text-to-image; one RGB PNG per prompt |
| Hub revision | `741f7c3ce8b383c54771c7003378a50191e9efe9` |
| Checkpoint format | Diffusers directory with `model_index.json`, transformer, VAE, both encoders/tokenizers, scheduler and safetensors weights |
| Schnell settings | Four steps, guidance 0, maximum sequence length 256 |
| License | Apache-2.0; review the hub model card and any download access requirements |

Schnell's guidance and sequence-length constraints follow the
[Diffusers FLUX documentation](https://huggingface.co/docs/diffusers/v0.37.0/en/api/pipelines/flux).
Single-file checkpoints, FLUX.1-dev, adapters, quantized checkpoints, and other FLUX families are outside this implementation.

## Recipe and performance

**Implementation draft: broader quality and hardware evaluation remain incomplete.** On October 8, 2026,
both Python and Dynamo variants passed correctness and completed benchmarks on one RTX 6000 Ada.
AITune SSIM scores were 0.724–0.938: above the provisional hard threshold of 0.70, with warnings below
the soft threshold of 0.95. Representative image quality/performance and a second GPU architecture still
require evaluation. [Recorded runs](results/README.md) describe the evidence and remaining gaps.

| Recipe | Intended execution | Precision | Compilation | Offloading | Deployment |
|---|---|---|---|---|---|
| [Single GPU](recipes/single-gpu.yaml) | One NVIDIA BF16-capable GPU, batch 1 | BF16 | Runtime transformer compilation; other components may use prebuilt backends | Disabled | Dynamo |

Local runs used one NVIDIA RTX 6000 Ada Generation GPU with 47.37 GiB reported VRAM. This is the only measured
architecture; minimum VRAM requirements have not been established. The full pipeline and tuning candidates must
fit on the selected device. Multi-GPU/context parallelism and Triton are not implemented.
The CLI rejects multiple visible GPUs and distributed launchers.

The transformer compares Torch-TensorRT JIT and TorchInductor JIT internally, with static shapes and the existing
FLUX reference's TensorRT partition settings. Other modules discovered by `aitune.torch.inspect` use AITune's default
AOT candidates, with maximum-batch-size discovery disabled. AITune selects each graph's throughput winner and may
retain eager execution. No quantization is requested. All loaded floating-point components use BF16; scheduler,
tokenizer, decoding, and backend accumulation operations may retain their native precision. Selected backend
descriptions and strategy settings are saved in `selection.json` and the raw tuning report.

Kernel Selector is not explicitly enabled by this recipe. Record any kernel behavior supplied by the selected
backend from the raw tuning report before publishing results. The internal transformer compiler settings follow
the [AITune FLUX example](https://github.com/ai-dynamo/aitune/tree/v0.6.0/examples/FLUX).
AITune's general API does not yet infer this complete FLUX recipe: model-specific candidate configuration remains
in `tune.py`. Users can change the supported workload settings without editing backend strategies, but this draft
does not establish the stronger claim of fully automatic recipe configuration.

Python results — single-gpu recipe, one RTX 6000 Ada, BF16, 1024×1024, four steps, batch/concurrency 1.
The October 8 run measured 20 images per variant, cycling over four bundled validation prompts with seed 42;
both variants passed the 0.70 hard correctness gate in fresh processes.
[Benchmark report](../../results/flux-schnell/single-gpu/benchmark/20261008T135335Z-49ad1389/report.json).

| Variant | Images/s | Mean latency (ms) | Peak GPU memory (GiB) | GPU utilization (%) | Compilation (s) | Warmup (s) |
|---|---:|---:|---:|---:|---:|---:|
| Original Model | 0.2981 | 3355.09 | 37.51 | 98.35 | 0.00 | 76.12 |
| AITune | 0.3867 | 2585.88 | 38.14 | 97.35 | 311.41 | 59.67 |

Memory and utilization use device-wide NVML samples every 50 ms during steady-state measurement.
Compilation is measured from compiler-frame intervals during loading and first use. First-use diagnostics were
23.90 s (Original Model) and 163.86 s (AITune); they can overlap compilation and are not added to it.
Warmup follows compilation. For this cold-cache comparison, verified references were reused and the same
isolated benchmark workers ran in a fresh container; each worker checked correctness before measurement.
The [exact invocation](../../results/flux-schnell/evaluation/20261008-tables/python-command.json) is retained.
The standard `recipe benchmark` command regenerates references first, which can populate TensorRT's timing cache.

Dynamo results — single-gpu recipe, one RTX 6000 Ada, BF16, 1024×1024, four steps, batch/concurrency 1,
image endpoint. On October 8, both services passed correctness and completed 20 AIPerf requests using
the same four prompts and seed 42. Reports: [Original Model](../../results/flux-schnell/single-gpu/deployment-benchmark-original/20261008T114920Z-05e1b0a5/report.json),
[AITune](../../results/flux-schnell/single-gpu/deployment-benchmark-aitune/20261008T120503Z-a0415d82/report.json).

| Variant | Images/s | Mean latency (ms) | Peak GPU memory (GiB) | GPU utilization (%) | Compilation (s) | Warmup (s) |
|---|---:|---:|---:|---:|---:|---:|
| Original Model | 0.3039 | 3288.71 | 37.52 | 92.64 | 0.00 | 76.10 |
| AITune | 0.3764 | 2654.99 | 38.15 | 92.17 | 316.98 | 59.25 |

Both benchmark reports embed fresh deployment correctness: Original Model SSIM was 1.0 for each sample;
AITune scores were 0.724–0.938, passing the 0.70 hard gate with warnings below 0.95. Server-side NVML samples
and request records were matched to each AIPerf measurement window, with no compilation during measured requests.
Compilation and warmup were measured in the service process before those requests. See
[measurement definitions](#measurement-and-quality-limits), [results status](results/README.md), and
[evaluation handoff](EVALUATION.md) for scope and remaining evaluation work. Report links require the local results tree.

## Environment and setup

The [Dockerfile](Dockerfile) uses the version-pinned NGC PyTorch 26.06 image and uv 0.12.17. A separate Python 3.12
virtual environment installs the [locked dependencies](uv.lock), rather than inheriting the image's Python packages.
The resolved stack includes AITune 0.6.0, PyTorch/Torch-TensorRT 2.12.1, TorchAO 0.17.0, TensorRT 10.16.1.11,
Diffusers 0.37.0, Transformers 4.57.6, Dynamo/runtime 1.5.0, and AIPerf 0.13.0.
The local Python and AITune Dynamo runs record these package versions, CUDA 13.0, and driver 580.173.02 on
RTX 6000 Ada. The local image ID and evaluation source snapshot are retained in the
[provenance record](../../results/flux-schnell/evaluation/20261008-tables/provenance.json). This locally built image
has no registry RepoDigest; complete image build provenance remains incomplete.
GPU commands record effective library, CUDA, cuDNN, driver, and device versions in their reports.

Sign in to Hugging Face and accept the access conditions on the
[model page](https://huggingface.co/black-forest-labs/FLUX.1-schnell). Create a
[Read token](https://huggingface.co/settings/tokens) for the same account.

Build and launch from this repository's root on the host. Enter the token at the prompt below;
`read -s` hides it and keeps its value out of shell history. If `HF_TOKEN` is already exported, skip the prompt.

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
  -v "$PWD/results/flux-schnell:/workspace/flux-schnell/results" \
  -v "$PWD/hf-cache:/cache/huggingface" \
  aitune-flux-schnell
```

The model loader pins the hub revision and uses `HF_TOKEN` automatically. The `-e HF_TOKEN` option passes the
exported host token into the container; no `hf auth login` command is needed. Do not put credentials in YAML,
source, reports, or the Dockerfile. The cache mount retains downloads.
For local checkpoints, add a read-only mount such as `-v /absolute/path/to/models:/models:ro`.

For evaluation, record `docker image inspect aitune-flux-schnell` outside the container and pass the evaluated
RepoDigest as `--env RECIPE_IMAGE_DIGEST=...`. Locally built images without a RepoDigest should retain their image ID
and build provenance alongside results; the report's missing digest remains explicit. No remote publication is needed.

## Default quickstart

Run inside the container:

```bash
uv run --frozen recipe tune --model-id black-forest-labs/FLUX.1-schnell --config recipes/single-gpu.yaml
uv run --frozen recipe correctness --model-id black-forest-labs/FLUX.1-schnell --config recipes/single-gpu.yaml
uv run --frozen recipe benchmark --model-id black-forest-labs/FLUX.1-schnell --config recipes/single-gpu.yaml
uv run --frozen recipe infer --model-id black-forest-labs/FLUX.1-schnell --config recipes/single-gpu.yaml
```

Every command requires the same explicit `--config`. Paths resolve relative to that file. The default artifact and
all persistent runs/caches are below the mounted `results/` directory. Each run prints its output location.

| Command | Output relative to the example directory | Expected result |
|---|---|---|
| tune | `results/single-gpu/artifact/tuned.ait` and `tuned.ait.json`; `results/single-gpu/tune/<run>/` | Artifact, source/workload/environment identity, SHA256, resolved config, backend selections and raw build/tuning records |
| correctness | `results/single-gpu/correctness/<run>/report.json`; `original/images/`, `aitune/images/` | Fresh separate processes generate reference/tuned PNGs; every image must meet the configured SSIM threshold (currently 0.70, provisional) |
| benchmark | `results/single-gpu/benchmark/<run>/report.json` | Both isolated variants, fresh-model correctness, per-image latency records, NVML samples and environment records |
| infer | `results/single-gpu/infer/<run>/aitune/images/` | One PNG per input ID, loaded from the saved artifact |

Benchmark runs correctness again before measurement and checks each newly loaded benchmark model against the saved
reference. Child stdout/stderr remains in each run's `original.log` and `aitune.log`; failures print the log path
and its last 30 lines. Artifact loading stages serialized tensors in CPU RAM before AITune restores the GPU
backends, avoiding a second full set of checkpoint weights in VRAM. Allow host RAM for those tensors in addition
to the loading process; inference still runs on the GPU with weight offloading disabled. Tuning reserves a new artifact
directory and refuses to overwrite it. For another tuning run, choose a new `artifact_path` parent in a copied recipe;
previous results and caches remain intact. Do not edit or replace an artifact while its service is running.

## Adapt the recipe

Copy the YAML, adjust supported values, and pass the new path to every command. Moving it requires adjusting relative
paths. The parser requires every section and rejects unsupported execution combinations.

| Setting | Supported change |
|---|---|
| `workload.width`, `height` | Multiples of 16, 256–2048; each recipe fixes one shape; fit and correctness require re-evaluation |
| `num_inference_steps`, `guidance_scale`, `max_sequence_length` | 1–4 steps, guidance 0, sequence length 1–256 |
| `inputs` | Separate tuning, validation, inference and benchmark JSONL manifests |
| `validation.threshold` | Hard SSIM acceptance threshold in (0, 1]; scores below it fail correctness |
| `validation.soft_threshold` | Optional warning threshold between the hard threshold and 1; omit or set null to disable |
| `benchmark` | Warmup passes, measured image count, NVML sample interval |
| `deployment` | Port, fixed seed, readiness timeout, image request count; concurrency remains 1 |
| `artifact_path`, `output_dir` | New artifact parent directory and persistent results location |
| `model.revision` | Exact 40-character commit of the compatible source passed with `--model-id` |

Each JSONL line contains exactly `id`, `prompt`, and an unsigned 32-bit `seed`:

```json
{"id":"lake","prompt":"A mountain lake at sunrise, landscape photograph.","seed":42}
```

IDs must be unique and contain only letters, numbers, `_` and `-`. Python generation honors each seed. Dynamo's image
adapter uses `deployment.seed` for all requests; validation and benchmark manifests must use that seed. To test
several seeds in deployment, create separate recipe configurations and restart the service between them.
The supplied prompts are small quickstart cases, not a representative evaluation suite.

Use `--checkpoint /models/my-schnell` in place of `--model-id` for a compatible complete Diffusers directory.
Local checkpoint files are SHA256-hashed for provenance. Hub sources pin `model.revision`. Changing source,
dimensions, denoising settings, precision, or execution invalidates the artifact contract and requires tuning again.
Loading checks the GPU compute capability and exact model/compiler library versions against the artifact build.
Artifact portability across drivers, CUDA versions, device models, or hosts is not established; use the same evaluated
image/GPU and preserve the complete results tree. Only load trusted AITune artifacts.

## Dynamo deployment

Both Original Model and AITune use the same Dynamo image protocol, GPU, image encoding, and workload settings.
The AITune service reloads the tuning artifact; the Original Model service loads the public checkpoint directly.
First complete the quickstart correctness command, then start one service:

```bash
uv run --frozen recipe serve --model-id black-forest-labs/FLUX.1-schnell \
  --config recipes/single-gpu.yaml --target dynamo --variant aitune
```

`serve` owns the frontend and worker, performs a complete first-use pass and warmup, and waits until its unique model
name appears in `/v1/models`. It prints `Ready` or fails on process exit/readiness timeout. Readiness can also be
inspected with `curl -fsS http://localhost:8000/v1/models`; the name and artifact identity are in
`results/single-gpu/serve-aitune/<run>/service.json`.

Open another shell in the same container (`docker exec -it aitune-flux-schnell bash`), then run:

```bash
cd /workspace/flux-schnell
uv run --frozen recipe deployment-correctness --config recipes/single-gpu.yaml \
  --target dynamo --variant aitune --endpoint http://localhost:8000
uv run --frozen recipe deployment-benchmark --config recipes/single-gpu.yaml \
  --target dynamo --variant aitune --endpoint http://localhost:8000
```

Stop with Ctrl-C in the serving shell. The launcher terminates and reaps both process groups. Start the baseline
with the identical `serve` command and `--variant original`, then run both deployment commands with
`--variant original`. Run the services sequentially. On forced container termination, keep the run logs for diagnosis.

The API is `POST /v1/images/generations`, with `model` from `/v1/models`, `prompt`, `n: 1`, configured `size` such as
`1024x1024`, and `response_format: b64_json`. The response contains one PNG in `data[0].b64_json`.
The task adapter fixes steps, guidance, text length, and seed from the recipe; it rejects different image dimensions
and batches. Current clients require a localhost base URL, the same results mount, and the same host clock.

Reports go to `results/single-gpu/deployment-correctness-<variant>/<run>/` and
`results/single-gpu/deployment-benchmark-<variant>/<run>/`. AIPerf receives the actual prompt manifest using its
`single_turn` loader and `image_generation` endpoint, with no synthetic chat inputs. Its command, inputs, logs and
raw artifacts are retained. The benchmark runs deployment correctness first and rejects incomplete/error runs,
unexpected compilation, or mismatched server/client request windows. See the
[AIPerf endpoint reference](https://docs.nvidia.com/aiperf/reference/command-line-options).

## Measurement and quality limits

SSIM is computed on lossless RGB output against the same-seed Original Model image with `data_range=255` and
`channel_axis=-1`. The per-image hard minimum (`validation.threshold`) is provisionally set to 0.70 for the current
evaluation, reduced from the initial 0.95 gate. The soft threshold (`validation.soft_threshold`) is 0.95:
scores below 0.70 fail correctness; scores from 0.70 up to (but excluding) 0.95 log a warning and pass;
scores at or above 0.95 pass without a warning. Python and Dynamo checks use the same behavior,
including checks invoked by benchmarks. New reports record both thresholds and per-image/overall
`soft_passed` alongside the hard-gate `passed`; soft fields are null when disabled. Changing the soft
threshold does not require retuning or invalidate saved hard-pass reference evidence; the next comparison
uses the new warning threshold. Justify both thresholds for your application. These are not evaluated
application-quality thresholds. SSIM detects structural changes but does not establish prompt alignment, aesthetics, or population-level
quality. Calibrate it with representative prompts, seeds, human review, and application-specific quality evidence.

Python latency is synchronized end-to-end prompt-to-PIL time, excluding PNG disk writes. Throughput is completed
images divided by the steady-state window. Dynamo latency includes HTTP and PNG/base64 work; GPU measurements use
the same AIPerf request window. Both paths report mean latency and concurrency 1. NVML samples whole-device memory
including non-PyTorch allocations every 50 ms, and integrates device-busy samples over that window. Peaks shorter
than the sampling interval can be missed. Use an otherwise idle GPU; measurements include other device users.

Every fresh process first executes all workload inputs outside measurement, then records configured warmup passes.
New timing records or changes in TorchDynamo graph counts or Inductor kernel counts during warmup/measurement
reject the report. Compilation time uses the union of PyTorch compiler-frame wall-time intervals collected
from before model loading through first use, including tracing, lowering, and synchronous backend builds.
Nested intervals are counted once; ordinary inference and prebuilt artifact loading are excluded. Raw intervals
are saved in `compilation.json`. This instrumentation is evaluated with the pinned PyTorch runtime compilation
path; it is not a general timer for arbitrary external compilers. Torch/Inductor/Triton/CUDA caches are isolated
per process under the run directory. The evaluated Python and Dynamo comparisons each used a fresh container
with an initially empty TensorRT timing cache; its default path is `/tmp/torch_tensorrt_engine_cache`.
Reusing a container can reuse that timing cache, so record its state when comparing compilation times.

First-use time is a separate diagnostic that includes generation and may overlap compilation; do not add it
to compilation as another preparation phase. Warmup is measured afterward. These runtime compilation values
exclude the original AOT artifact builds and tuning/search, whose raw records remain in the
[tuning report](../../results/flux-schnell/single-gpu/tune/20261001T151244Z-80180140/report.json).
Zero denotes observed absence of new compiler-frame work in the Original Model eager process. Older reports
without this instrumentation retain null compilation values rather than retroactively estimating them.

The recipe is excluded from the ready catalog and docs navigation until fresh-artifact correctness, representative
quality/performance measurements, the effective environment, and at least two GPU architectures are documented.
