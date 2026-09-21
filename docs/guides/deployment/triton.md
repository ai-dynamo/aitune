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
        ├── fast.yaml
        └── manual.yaml
```

AITune copies the executable and any required additional files, preserving their relative paths. ONNX models with
external data use a directory beneath the version entry. The generated `config.pbtxt` describes the artifact's
inputs, outputs, and runtime settings. AITune also generates Model Analyzer configurations and includes representative
input data when the backend retained a sample.

Publication stages all files outside the repository and moves the completed directory into place. A failure therefore
does not leave a partially published model. Publication never replaces an existing model directory.

## Serve the repository

Choose one of the following ways to serve the repository. Repository generation is the same for both.

### Run Triton through Dynamo

Dynamo adds service discovery and routing to the Triton repository. Triton still loads and
executes the model. Requests use the KServe gRPC protocol rather than Dynamo's OpenAI-compatible HTTP API.

Prepare an environment containing both Dynamo and Triton using the
[official Dynamo Triton documentation](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/backends/triton/overview)
for the supported release and container setup. Installing `aitune[dynamo]` alone does not provide a Triton server.

Mount the local `model_repository/` directory at `/models` in that environment. The following commands run inside
that environment. Start Dynamo's KServe gRPC frontend:

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

In an environment with Triton and the required backend installed, start Triton against the repository root:

```bash
tritonserver --model-repository="$PWD/model_repository"
```

You can run the same command in an NVIDIA Triton container by mounting `model_repository` at a path inside the
container. The command's path must refer to that mount inside the container. Use a Triton release whose backend and
CUDA versions are compatible with the exported artifact and deployment system.

## Use a Python model instead

Choose Triton's Python backend when your deployment needs Python model construction, preprocessing, or pipeline code.

1. [Save an AITune checkpoint](checkpoints.md#save-after-tuning) after tuning.
2. Create a Triton Python model repository with a `config.pbtxt` and a versioned `model.py`.
3. In `TritonPythonModel.initialize()`, construct the original model and call `aitune.torch.load()`.
4. In `execute()`, convert each Triton request into model inputs, run inference, and return Triton responses.

You provide the repository and Python code for this path; `publish()` generates repositories only for the supported
ONNX, TensorRT, and PT2 artifacts. Install AITune, the tuned backend's dependencies, and the model's Python dependencies
in the Python backend environment. Make the checkpoint available at the configured storage path.

Follow the [Triton Python backend guide](https://github.com/triton-inference-server/python_backend#usage) for its
repository layout, configuration, and request-handling API. Once prepared, the repository can be served with
[standalone Triton](#run-triton-standalone) or a [Dynamo Triton environment](#run-triton-through-dynamo) that includes
the Python backend and those dependencies.

## Optional batching overrides

The basic publication call keeps the model's full tensor shapes and leaves Triton's implicit and dynamic batching
disabled. No batch limit needs to be supplied for that call.

Enable `dynamic_batching` when Triton should combine independent requests. AITune reads the supported batch range from
the tuned artifact and uses its recorded maximum, so the deployment limit normally does not need to be provided:

```python
model_path = publish(
    artifact,
    path="model_repository",
    model_name="encoder",
    dynamic_batching=True,
)
```

Set `max_batch_size` only to lower the deployment limit from the recorded maximum. It can also enable Triton's implicit
batch dimension without enabling the dynamic batcher. The override must stay within the artifact's recorded bounds,
and every input and output must use the first axis as its batch dimension.

TensorRT model configurations retain the complete optimization-profile set so Triton can select a compatible profile
for each request.

## Profile with Model Analyzer

Publication generates two configurations under `model_analyzer/`:

- `fast.yaml` provides a smaller search for quick feedback.
- `manual.yaml` provides a larger explicit sweep over request batch size, model maximum batch size, request concurrency,
  instance count, and dynamic-batching queue delay.

The configurations use the artifact's recorded minimum input shapes. Batched deployments omit the leading batch
dimension from Perf Analyzer shape flags. Model Analyzer skips combinations where the Perf Analyzer request batch size
exceeds the candidate model maximum.

TensorRT optimization-profile shape bounds do not constrain the search. For plans with multiple profiles, the complete
profile set remains static while Model Analyzer varies Triton settings such as instance count.

When a backend retained representative tuning inputs, publication also writes `model_analyzer/input-data.json` and
configures Perf Analyzer to use those values. This is important for embeddings and other inputs whose values must stay
within a valid domain. Batched samples are reduced to one example and resized to the profiling shape by repeating or
truncating recorded values.

AITune generates the configuration but does not run Model Analyzer. Checkpoints, results, and generated repositories
are written under `model_repository-model-analyzer/encoder/`, outside the serving repository. See the
[Model Analyzer configuration reference](https://github.com/triton-inference-server/model_analyzer/blob/main/docs/config.md)
for commands and additional options.

## Known limitations

Artifact publication exports one compiled module graph. It does not package surrounding Python preprocessing,
postprocessing, or a complete pipeline automatically. Keep that code in the application, or use the Python backend
when it needs to run inside Triton.

For Dynamo's Triton feature support and limitations, see the
[official Dynamo guide](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/backends/triton/overview).

## Related guides

- [Deploy a Python model with Dynamo](dynamo.md)
- [Save and load checkpoints](checkpoints.md)
