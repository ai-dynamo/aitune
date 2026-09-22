---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Deploy with Dynamo"
---

Start by serving FLUX image generation on one GPU with NVIDIA Dynamo. The walkthrough below loads a tuned checkpoint,
starts the frontend and worker, and sends an image request. For video generation across multiple GPUs, continue with
the [WAN example in the multi-GPU section](#multi-gpu-workers).

Your application supplies the inference callable and input conversion. AITune's Dynamo worker registers the endpoint
and converts supported outputs into responses.

If you want Dynamo to serve an exported ONNX, TensorRT, or PT2 model, follow
[Run Triton through Dynamo](triton.md#run-triton-through-dynamo). That path uses a Triton repository and KServe gRPC.

## Serve FLUX on one GPU

This walkthrough uses the existing FLUX worker and client from the AITune repository. Run the commands in the
[FLUX example environment](../../../examples/FLUX/README.md#environment-setup), with access to FLUX.1-dev and a GPU
with enough memory for the pipeline.

### 1. Prepare the tuned checkpoint

From the repository root, enter the example directory and install its Dynamo dependencies:

```bash
cd examples/FLUX
uv pip install ".[dynamo]"
```

If you have not tuned the model yet:

```bash
uv run tune
```

With the default settings, this saves `checkpoints/flux-dev.ait`. In `config.yaml`, keep
`Backend.tuned_model_path: "flux-dev.ait"` and the generation settings used for tuning. The worker constructs the
pipeline and loads this checkpoint during startup. See [Save and load checkpoints](checkpoints.md) for custom paths.

### 2. Start the frontend

Open a terminal in `examples/FLUX` and start Dynamo's HTTP frontend:

```bash
DYN_DISCOVERY_BACKEND=file uv run python -m dynamo.frontend --http-port 8000
```

File-based discovery lets the local frontend and worker find each other without running etcd. Leave this process
running.

### 3. Start the FLUX worker

Open a second terminal in `examples/FLUX` and start the worker:

```bash
DYN_DISCOVERY_BACKEND=file uv run python -m flux.dynamo.backend
```

The worker loads and warms up the tuned pipeline, then registers it with Dynamo. Wait for the registration message
before sending a request.

### 4. Generate an image

From a third terminal in `examples/FLUX`, run the client:

```bash
uv run python -m flux.dynamo.client --prompt "A futuristic city at sunset"
```

The client sends a request to `/v1/images/generations` and saves the result as `output.png`. Send more requests with
different prompts using the same command. The frontend and worker keep running until you stop them. The example
processes one generation request at a time.

## Multi-GPU workers

Use the [WAN video generation example](../../../examples/WAN/README.md) to serve a pipeline across multiple GPUs.
It uses Diffusers context parallelism, with every rank participating in each generation request.

### Prepare WAN on four GPUs

Follow the [WAN environment setup](../../../examples/WAN/README.md#environment-setup) and
[four-GPU tuning instructions](../../../examples/WAN/README.md#tune-on-four-gpus) first. Keep the same GPU count,
context-parallel mode, and generation shapes for serving. Each rank loads its own checkpoint.

From the repository root, enter `examples/WAN` and install the serving dependencies:

```bash
cd examples/WAN
uv pip install ".[dynamo]"
```

Check `config.yaml` in that directory. WAN's settings are at the top level: `tuned_model_path`, `context_parallel`,
`height`, `width`, and `num_frames` must match the tuning run.

### Start the frontend and WAN workers

Use a separate local session for WAN; stop the FLUX worker and frontend first if you ran the single-GPU walkthrough.

From `examples/WAN`, [start the frontend](#2-start-the-frontend) with the command above. In another terminal in
`examples/WAN`, launch one worker per GPU:

```bash
DYN_DISCOVERY_BACKEND=file uv run torchrun --standalone --nproc-per-node=4 \
  --module wan.dynamo.backend
```

Wait for model registration, then send a video request from a third terminal in `examples/WAN`:

```bash
uv run python -m wan.dynamo.client --prompt "A white ferret swimming in a mountain stream"
```

The client calls `/v1/videos` and saves `output.mp4`. For a launcher that starts the service, sends one request, and
shuts it down, see the [WAN serving example](../../../examples/WAN/README.md#serve-with-nvidia-dynamo).

## Worker configuration

The callable passed to `dynamo_worker()` is the model implementation that AITune executes. The application creates
and loads the model before serving requests, either before starting the worker or in its `setup` callback.

The name `model_path` comes from Dynamo's `register_model()` API. AITune passes this value to Dynamo as a registration
reference; AITune does not read it or use it to create the callable. `model_name` is an optional public alias that
clients use in requests. Keeping them separate lets a deployment use one reference for registration and expose a
stable name such as `production-flux`. When `model_name` is omitted, AITune uses `model_path` as the public name.

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `str` | required | Modality: `"image"`, `"video"`, `"audio"` for text to speech, or `"embedding"` |
| `model_path` | `str` | required | Registration reference passed unchanged to Dynamo; it is not used to create or load the callable |
| `mapping` | `Callable \| None` | `None` | Converts a request to keyword arguments for the callable; required for an `nn.Module` |
| `namespace` | `str` | `"aitune"` | Dynamo service namespace |
| `component` | `str` | `"backend"` | Component name within the namespace |
| `endpoint` | `str` | `"generate"` | Endpoint name; the full address is `{namespace}.{component}.{endpoint}` |
| `enable_nats` | `bool` | `False` | Enables NATS JetStream for KV-cache events |
| `model_name` | `str \| None` | `None` | Optional public model name advertised to clients; defaults to `model_path` |

## Supported modalities

| `type` | Request field | Expected return value | Example |
|---|---|---|---|
| `"image"` | `request.prompt` | PNG or JPEG bytes | FLUX, Stable Diffusion |
| `"video"` | `request.prompt` | MP4 bytes | WAN |
| `"audio"` | `request.input` | WAV bytes | Qwen3-TTS |
| `"embedding"` | `request.input` as a string or list of strings | `np.ndarray` or `torch.Tensor` with shape `(n, dim)` | E5Large, BGE |

Audio bytes are packed as WAV output. To serve another audio codec, return a complete Dynamo response dictionary.
Plain dictionaries are forwarded to the runtime unchanged for every modality.

## Customize worker startup

Use `setup` to initialize the model at worker startup and `warmup` to exercise it before endpoint registration:

```python
dyn.dynamo_worker(model_or_fn, config, setup=setup, warmup=warmup)
```

The callable can also use a model loaded before worker startup. A plain callable receives the deserialized
request when `mapping` is omitted; an `nn.Module` requires a mapping to its input arguments.

For full lifecycle control, subclass `DynamoWorker` and implement `setup()` and `serve()`. The base class also
provides `warmup()` and `on_ready(runtime, endpoint)` hooks. With this lower-level API, your `on_ready`
implementation must register the model with the frontend, and `serve` must yield responses in Dynamo's wire format.
The functional `dynamo_worker()` API handles that registration and response conversion for you.

## Adapt your own multi-GPU worker

Use the same worker API for single-GPU and multi-GPU models. The application initializes the process group before
calling `run()` or `dynamo_worker()` on every rank. Construct the rank-local model before the call or in `setup`.
AITune starts the Dynamo endpoint only on rank 0, broadcasts each request, runs inference on every rank, and returns
only rank 0's response.

Pass rank-local initialization that may fail through `setup=` and collective model warmup through `warmup=`. AITune
exchanges setup failures across the process group before any rank starts warmup or registers the endpoint.

AITune does not initialize or destroy the process group, add inference barriers, or synchronize CUDA. The application
owns device placement, model sharding or context parallelism, and process-group cleanup. One initialized multi-rank
process group represents one collective model worker. Run separate worker groups or pods for independent replicas.

## Known limitations

AITune's `DynamoWorker` has the following limitations:

- The `"audio"` modality requires NVIDIA Dynamo 1.1 or later and serves text-to-speech requests through
  `/v1/audio/speech`.
- Automatic speech recognition is not currently exposed by the worker API.

## Next steps

- [FLUX image generation](../../../examples/FLUX/README.md)
- [WAN video generation](../../../examples/WAN/README.md)
- [Deploy through Triton](triton.md)
- [Deployment overview](deployment.md)
