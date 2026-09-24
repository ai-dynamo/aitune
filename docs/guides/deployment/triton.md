---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Deploy with Triton"
---

Use Triton within Dynamo to execute your model behind Dynamo's frontend and routing. AITune can generate a model
repository that Triton loads directly. The same repository can also run on standalone Triton.

This guide starts with repository generation for ONNX Runtime, TensorRT, and TorchInductor AOT artifacts. If you need
to retain Python model or pipeline code, go to [Use a Python model instead](#use-a-python-model-instead).

## Generate a model repository

Install the Triton publishing dependencies:

```bash
uv pip install "aitune[triton]"
```

After tuning, retrieve the artifact from the wrapped module and publish it. Here, `model` is the AITune-wrapped
module with one compiled graph, using ONNX Runtime, TensorRT, or TorchInductor AOT:

```python
from aitune.triton import publish

artifact = model.artifact()
model_path = publish(
    artifact,
    path="model_repository",
    model_name="encoder",
)
```

`path` is the repository root; `model_name` is the name clients will request. This call returns
`model_repository/encoder`. The model directory contains:

```text
model_repository/
└── encoder/
    ├── 1/
    │   └── model.onnx  # model.plan or model.pt2 for other runtimes
    ├── config.pbtxt
    └── model_analyzer/
        └── config.yaml
```

AITune copies the executable and any required additional files, preserving their relative paths. ONNX models with
external data use a directory beneath the version entry. The generated `config.pbtxt` describes the artifact's
inputs, outputs, and runtime settings. AITune also generates a Model Analyzer configuration and includes representative
input data when the backend retained a sample.

### Publish an existing model file

For a model that was not tuned by AITune, provide a backend-specific configuration with the tensor interface and
batching settings. The config type selects the Triton backend; AITune does not inspect or compile the file:

```python
from aitune.triton import ONNXRuntimeModelConfig, TritonDataType, TritonTensorConfig, publish

config = ONNXRuntimeModelConfig(
    name="encoder",
    max_batch_size=8,
    execution_provider="cuda",
    inputs=(TritonTensorConfig(name="input", data_type=TritonDataType.FLOAT16, dims=(3, 224, 224)),),
    outputs=(TritonTensorConfig(name="output", data_type=TritonDataType.FLOAT16, dims=(1000,)),),
)
model_path = publish("encoder.onnx", path="model_repository", config=config)
```

The default `batcher=DynamicBatcher()` combines independent requests. For a stateful model, set
`batcher=SequenceBatcher(...)` instead. Set `batcher=None` to omit both schedulers; a positive `max_batch_size` still
allows batched requests, while `max_batch_size=0` disables batching and requires `batcher=None`.

Use `additional_files` for ONNX external data and `resources` for referenced labels or warmup files. Existing-file
publication writes `config.pbtxt` but has no artifact from which to derive Model Analyzer input shapes or samples.

`publish()` generates files directly in the new model directory; it does not make a live deployment update. It never
replaces an existing model directory. If generation fails or is interrupted, an incomplete directory can remain and
must be removed or archived before retrying. Generate into a repository that Triton is not watching, then make the
completed repository available using your deployment process and Triton's
[model-control policy](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_management.md).
Triton's `poll` mode can observe incomplete changes, so do not rely on it to protect an in-progress generation.

## Serve the repository

Choose one of the following ways to serve the repository. Repository generation is the same for both.

Choose an image whose Triton, backend, CUDA, and NVIDIA driver versions are compatible with the system where the model
was exported. The setup for each deployment option is described below.

### Run Triton through Dynamo

Dynamo adds service discovery and routing to the Triton repository. Triton still loads and
executes the model. Requests use the KServe gRPC protocol rather than Dynamo's OpenAI-compatible HTTP API.

Prepare an environment containing both Dynamo and Triton using the
[Dynamo Triton container quick start](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/backends/triton/overview#quick-start-prebuilt--release-container).
It explains how to build the combined Dynamo and Triton image and start it with NVIDIA GPU access.

When starting the container, add `-v "$PWD/model_repository:/models"` to mount the generated repository at `/models`.
The following commands run inside that container. Start Dynamo's KServe gRPC frontend:

```bash
python3 -m dynamo.frontend \
  --kserve-grpc-server \
  --http-port=8001 \
  --discovery-backend=file &
```

Then start Dynamo's Triton worker:

```bash
python3 -m dynamo.triton \
  --model-repository=/models \
  --discovery-backend=file
```

The worker registers every model in the repository with Dynamo. Send requests to port `8001` with a KServe gRPC
client and the published model name. A single worker can serve multiple models, and Dynamo routes requests across
registered workers. The repository contents are the same as for standalone Triton.

### Run Triton standalone

The [Triton quick start](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html)
explains the container prerequisites and release selection. After choosing a compatible release, start the server with
the generated repository mounted at `/models`:

```bash
docker run --gpus all --rm \
  --shm-size=1g \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v "$PWD/model_repository:/models" \
  nvcr.io/nvidia/tritonserver:<release>-py3 \
  tritonserver --model-repository=/models
```

Replace `<release>` with a Triton release such as `YY.MM`. Ports `8000`, `8001`, and `8002` expose Triton's HTTP,
gRPC, and metrics endpoints. If Triton is already installed in the current environment, run
`tritonserver --model-repository="$PWD/model_repository"` directly instead.

## Use a Python model instead

Choose Triton's Python backend when your deployment needs Python model construction, preprocessing, or pipeline code.

1. [Save an AITune checkpoint](checkpoints.md#save-after-tuning) after tuning.
2. Create a Triton Python model repository with a `config.pbtxt` and a versioned `model.py`.
3. In `TritonPythonModel.initialize()`, construct the original model and call `aitune.torch.load()`.
4. In `execute()`, convert each Triton request into model inputs, run inference, and return Triton responses.

You provide the repository and Python code for this path; `publish()` generates repositories for supported
ONNX, TensorRT, and PT2 artifacts or existing model files. Install AITune, the tuned backend's dependencies, and the model's Python dependencies
in the Python backend environment. Make the checkpoint available at the configured storage path.

Follow the [Triton Python backend guide](https://github.com/triton-inference-server/python_backend#usage) for its
repository layout, configuration, and request-handling API. Once prepared, the repository can be served with
[standalone Triton](#run-triton-standalone) or a [Dynamo Triton environment](#run-triton-through-dynamo) that includes
the Python backend and those dependencies.

## Artifact batching

AITune derives Triton's batching configuration from the artifact. When every input and output has a batch axis on the
first dimension, the generated `max_batch_size` is the smallest recorded maximum across those tensors. A batch-one
artifact keeps `max_batch_size=1` and the implicit batch dimension, without a dynamic batcher. Dynamic batching is
enabled when the supported maximum is at least 2. Artifacts without a compatible batch axis use `max_batch_size=0`
and retain their full tensor shapes. Structured PT2 calls also use `max_batch_size=0`.

For an existing model file, supply the batching settings through its backend-specific `config`. Its `max_batch_size`
and `batcher` fields control Triton's implicit batch dimension and scheduler.

TensorRT model configurations retain the complete optimization-profile set so Triton can select a compatible profile
for each request.

## Profile with Model Analyzer

Publication generates `model_analyzer/config.yaml` for a bounded search. Perf Analyzer concurrency is capped at twice
the published maximum batch size.

The configuration uses the artifact's recorded minimum input shapes. Batched deployments omit the leading batch
dimension from Perf Analyzer shape flags. Model Analyzer skips combinations where the Perf Analyzer request batch size
exceeds the candidate model maximum.

TensorRT optimization-profile shape bounds do not constrain the generated quick search. Users can copy and modify the
generated configuration when they need a custom search space.

When a backend retained representative tuning inputs, publication also writes `model_analyzer/input-data.json` and
configures Perf Analyzer to use those values. This is important for embeddings and other inputs whose values must stay
within a valid domain. Batched samples are reduced to one example and resized to the profiling shape by repeating or
truncating recorded values.

AITune generates the configuration but does not run Model Analyzer. Checkpoints, results, and generated repositories
are written under `model_repository-model-analyzer/encoder/`, outside the serving repository. See the
[Model Analyzer configuration reference](https://github.com/triton-inference-server/model_analyzer/blob/main/docs/config.md)
for commands and additional options.

For an explicit bounded sweep, `generate_model_analyzer_configs(artifact, model_path=model_path, path=...)` writes
`fast.yaml` and `manual.yaml` to a separate directory. Use this only when you need to customize the search space;
the default Triton workflow uses `model_analyzer/config.yaml`.

## Known limitations

Artifact publication exports one compiled module graph. It does not package surrounding Python preprocessing,
postprocessing, or a complete pipeline automatically. Keep that code in the application, or use the Python backend
when it needs to run inside Triton.

For Dynamo's Triton feature support and limitations, see the
[official Dynamo guide](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/backends/triton/overview).

## Related guides

- [Deploy a Python model with Dynamo](dynamo.md)
- [Save and load checkpoints](checkpoints.md)
