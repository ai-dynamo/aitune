---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Deployment Guide"
---

This guide covers the full deployment story for AITune-tuned models: saving a tuned model to a checkpoint, loading it in production, and optionally serving it as an OpenAI-compatible HTTP endpoint via [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo).

## Publish an Existing Model to Triton

Install `aitune[triton]` to publish an ONNX graph, TensorRT plan, or AOTInductor PT2
package that you already have. No AITune tuning flow is required.

```python
from aitune.triton import ONNXRuntimeModelConfig, publish

config = ONNXRuntimeModelConfig(
    name="encoder",
    execution_provider="cuda",
    max_batch_size=8,
    dynamic_batching=True,
    inputs=({"name": "input_ids", "data_type": "TYPE_INT64", "dims": (-1,)},),
    outputs=({"name": "embedding", "data_type": "TYPE_FP32", "dims": (768,)},),
)

model_directory = publish(
    "models/encoder.onnx",
    path="model_repository",
    config=config,
    model_version=1,
)
```

This writes `model_repository/encoder/config.pbtxt` and
`model_repository/encoder/1/model.onnx`. The configuration supplies the tensor
names, types, shapes, and batching contract; these must match your model. When
`max_batch_size` is positive, `dims` excludes the leading batch dimension. Use
`max_batch_size=0` and full tensor dimensions for an unbatched model.

Choose `TensorRTModelConfig` with `optimization_profile_indices` and optional
`cuda_graphs` for a TensorRT plan, or `TorchAOTIModelConfig` with `structured_call`
for a PT2 package. The configuration type selects the backend and destination
filename (`model.plan`, `model.onnx`, or `model.pt2`).

For ONNX external data, pass `companions=["weights.bin", "data/weights.bin"]`
with the files your graph actually references. Paths are relative to the source
graph's directory. The graph and companions are copied under `1/model.onnx/`,
preserving the companion paths. Files are staged before publication, and an
existing model directory is never replaced. This API prepares the repository;
it does not start Triton or verify that the model can execute on the target hardware.

Tuned artifacts continue to use `publish(artifact, path=..., model_name=...)`,
with configuration derived from the artifact.

## Save a Tuned Model

### Basic Save

```python
import aitune.torch as ait

# After tuning
ait.save(model, "checkpoints/model.ait")
```

This creates:

- `checkpoints/model.ait`: Compressed checkpoint with tuned modules
- `checkpoints/model_sha256_sums.txt`: SHA256 checksums
- `checkpoints/model/`: Decompressed artifacts (after first load)

### With Custom Storage

```python
from aitune.torch import LocalTorchStorage

storage = LocalTorchStorage(
    base_folder="production/models",
    remove_checkpoint_after_tune=False,
)

ait.save(model, "model_v2.ait", storage=storage)
```

## Load in Production

### Basic Load

```python
import aitune.torch as ait

model = YourModel()
model.eval()
model.to("cuda")

ait.load(model, "checkpoints/model.ait")

output = model(input_data)
```

### With Custom Storage

```python
from aitune.torch import LocalTorchStorage

storage = LocalTorchStorage(base_folder="production/models")
ait.load(model, "model.ait", storage=storage)
```

### Loading Process

1. **First load** — decompresses `.ait` file, extracts artifacts, verifies checksums, loads backend and weights. Slower due to decompression.
2. **Subsequent loads** — uses decompressed files from `checkpoints/`, skips decompression. Faster startup.

## Serve with Dynamo Worker

After loading a tuned model, you can expose it as an OpenAI-compatible HTTP endpoint using AITune's Dynamo integration. The worker registers the model with the Dynamo HTTP frontend, deserializes incoming requests, packs inference results into the Dynamo wire format, and blocks until SIGTERM/SIGINT.

### Prerequisites

Install the Dynamo extra:

```bash
uv pip install "aitune[dynamo]"
```

The `"audio"` modality requires NVIDIA Dynamo 1.1 or later and serves text-to-speech requests through
`/v1/audio/speech`. Automatic speech recognition is not currently exposed by this worker API.

For local development without etcd/NATS, set `DYN_DISCOVERY_BACKEND=file` before starting any Dynamo process.

### Quick Start — Embedding Model

```python
import numpy as np
import aitune.dynamo as dyn
import aitune.torch as ait  # for ait.config, ait.load, ait.save
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/e5-large-v2")
ait.config.device_after_tuning = "cpu"
ait.load(model, "checkpoints/e5large.ait")


def mapping(req) -> dict:
    sentences = req.input if isinstance(req.input, list) else [req.input]
    return {"sentences": sentences}


def embed(sentences: list[str]) -> np.ndarray:
    return model.encode(sentences, normalize_embeddings=True, device="cuda")


config = dyn.DynamoWorkerConfig(
    type="embedding",
    model_path="intfloat/e5-large-v2",
    mapping=mapping,
)
dyn.dynamo_worker(embed, config)  # blocks until shutdown
```

### API Reference

Import from `aitune.dynamo`:

```python
import aitune.dynamo as dyn
```

#### `DynamoWorkerConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `str` | required | Modality: `"embedding"`, `"image"`, `"video"`, or `"audio"` (TTS) |
| `model_path` | `str` | required | HuggingFace model ID or local path |
| `mapping` | `Callable \| None` | `None` | Adapter `fn(request) -> dict` unpacked as `**kwargs` into the user function. Required when passing an `nn.Module`. |
| `namespace` | `str` | `"aitune"` | Dynamo service namespace |
| `component` | `str` | `"backend"` | Component name within the namespace |
| `endpoint` | `str` | `"generate"` | Endpoint name — full address: `{namespace}.{component}.{endpoint}` |
| `enable_nats` | `bool` | `False` | Enable NATS JetStream for KV-cache events |
| `model_name` | `str \| None` | `None` | Name advertised to the frontend. Defaults to `model_path`. |

#### `dynamo_worker(model_or_fn, config, *, setup=None, warmup=None)`

Functional API. Validates config, starts the Dynamo runtime, and blocks until shutdown.

- **`model_or_fn`**: any callable, or a `torch.nn.Module` (requires `config.mapping`)
- **`config`**: `DynamoWorkerConfig`
- **`setup`**: optional rank-local initialization callback
- **`warmup`**: optional warmup callback run only after setup succeeds on every rank

#### `DynamoWorker` (class-based API)

For more control, subclass `DynamoWorker` and override `setup()` and `serve()`:

```python
import aitune.dynamo as dyn


class MyEmbeddingWorker(dyn.DynamoWorker):
    def setup(self) -> None:
        # called once at startup — load or tune your model here
        self.model = load_my_model()

    async def serve(self, request):
        sentences = request.input if isinstance(request.input, list) else [request.input]
        embeddings = self.model.encode(sentences)
        yield embeddings


MyEmbeddingWorker().run()
```

Override `on_ready(runtime, endpoint)` for post-startup work such as custom `register_model` calls.
Override `warmup()` to run model warmup after `setup()` succeeds on every rank and before endpoint registration.

### Multi-GPU Workers

Use the same `DynamoWorker` API for local and multi-GPU models. Initialize the application-owned process group and
construct the rank-local model before calling `run()` or `dynamo_worker()` on every rank. AITune starts the Dynamo
endpoint only on rank 0, serializes and broadcasts incoming requests, executes the request on every rank, and returns
only rank 0's response.

For rank-local initialization that may fail, pass it through `setup=` and put any collective model warmup in
`warmup=`. AITune exchanges setup failures across the process group before allowing any rank to start warmup or
register the endpoint.

The worker does not initialize or destroy the process group and does not add inference barriers or CUDA
synchronization. The application remains responsible for device placement, model sharding or context parallelism,
and process-group teardown after the worker returns. An initialized multi-rank process group represents one collective
model worker. Deploy independent model replicas as separate worker groups or pods.

### Modality Types

| `type` | Request field | Expected return type | Example |
|---|---|---|---|
| `"embedding"` | `request.input` (str or list[str]) | `np.ndarray` or `torch.Tensor` of shape `(n, dim)` | E5Large, BGE |
| `"image"` | `request.prompt` (str) | PNG or JPEG `bytes` | FLUX, Stable Diffusion |
| `"video"` | `request.prompt` (str) | MP4 bytes | — |
| `"audio"` | `request.input` (str) | WAV bytes | Qwen3-TTS |

Audio bytes are automatically packed as WAV output. To serve another audio codec, return a fully formed Dynamo response
`dict`. Plain dictionaries are forwarded to the runtime as-is for every modality.

### Serving with `run_dynamo.sh`

The recommended way to start all processes locally is a `run_dynamo.sh` script that:

1. Starts the Dynamo HTTP frontend in the background
2. Starts the backend worker in the background
3. Polls `/health` until the endpoint is registered
4. Runs a smoke-test client request

```bash
#!/bin/bash
export DYN_DISCOVERY_BACKEND=file

python -m dynamo.frontend --http-port 8000 &
FRONTEND_PID=$!

python -m myapp.dynamo.backend &
BACKEND_PID=$!

trap "kill -9 $FRONTEND_PID; kill -9 $BACKEND_PID" EXIT

for i in {1..10}; do
  curl -s http://localhost:8000/health | grep -q '"dyn://aitune.backend.generate"' && break
  echo "Waiting for endpoint... (attempt $i)"
  sleep 5
done

python -m myapp.dynamo.client
```

See the [E5Large example](../../../examples/E5Large/README.md) for a complete working version.

## Next Steps

- [AOT Tuning Guide](../aot_tuning.md) — tuning a model before saving
- [Tune Strategies](../tune_strategies/tune_strategies.md) — selecting the right optimization strategy
- [Backend Guides](../backends/tensorrt_backend.md) — backend-specific deployment notes
- [E5Large example](../../../examples/E5Large/README.md) — end-to-end embedding worker
- [FLUX example](../../../examples/FLUX/README.md) — end-to-end image generation worker
