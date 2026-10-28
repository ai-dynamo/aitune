---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---
# AGENTS.md: tests/functional_ci

Fast GPU functional tests for the AITune public API. Most tests tune a small seeded toy model with a backend from
`BACKENDS` and compare the output with eager PyTorch. Strategy and kernel-selector tests build their own backends.

```bash
pytest -rfEs tests/functional_ci                                        # whole suite, needs CUDA
pytest tests/functional_ci/test_shapes.py -k tensorrt                   # one file, backend ids with "tensorrt"
AITUNE_MATRIX_RESULTS=matrix.json pytest tests/functional_ci            # also write the feature-matrix JSON
```

CI runs the suite in the `Functional CI tests` step of the `Functional CI GPU` job in `.github/workflows/ci.yml`.
The suite fails (does not skip) when CUDA is not available.

## Rules

- Use only plain `torch` toy models: no diffusers, Hugging Face, or other model libraries, and no weight downloads.
- Keep toys realistic. If a toy finds a backend gap that also affects real models, fix AITune, not the toy.
- Give each test a `@pytest.mark.matrix(...)` marker for the feature-matrix cells that it proves.

Backend tests (tests that use `BACKENDS`) also follow these rules:

- Pin the backend with `pinned_strategy(backend)` from `lifecycle.py`.
- Compare with eager through `run_lifecycle` or `assert_matches_eager` and the backend tolerance.
- Write one test per backend group or outcome. Select the group with `with_backends(...)`, not an `if` in the body.

`test_strategies.py` and `test_kernel_selector.py` test strategy selection and `KernelSelectorBackend`, so they build
their own strategies.

## Files

| File | Content |
| --- | --- |
| `backends.py` | `BACKENDS` table, `Capability` names, `with_backends()` |
| `lifecycle.py` | `Tolerance`, `TENSORRT_FP32`, `pinned_strategy`, `run_lifecycle`, `assert_matches_eager`, `selected_backend` |
| `models.py` | Seeded toys: `make_mlp`, `make_image_model`, `make_input`, `seeded` |
| `flux.py` | Toy Flux pipeline (`ToyFluxPipeline`): CLIP/T5-like encoders, MMDiT, VAE decoder |
| `artifacts.py` | Runs exported artifacts (`pt2`, `tensorrt_plan`, `onnx`) in their native runtime, without AITune |
| `conftest.py` | `device` fixture, cleanup after each test, `matrix` marker and JSON writer |
| `test_*.py` | One file per feature area: lifecycle, shapes, topology, JIT, strategies, precision, ONNX input, pipelines, kernel selector |

## Backends

`BACKENDS` in `backends.py` maps a backend id to a `BackendCase`:

- `factory`: builds a new `Backend`.
- `tolerance`: allowed difference from eager.
- `capabilities`: features that the backend is tested on.
- `artifact_format`: deployable format; set only for backends with `export`.

The backend id is the pytest parametrize id and the feature-matrix column.

| Backend id | Backend class | Tolerance | Capabilities |
| --- | --- | --- | --- |
| `torch-eager` | `TorchEagerBackend` | default | `save_load` |
| `inductor-aot` | `TorchInductorAotBackend` | default | `save_load`, `export` (`pt2`), `dynamic_batch`, `dynamic_non_batch`, `structured_io`, `multigraph` |
| `torch-tensorrt-aot` | `TorchTensorRTAotBackend` | `TENSORRT_FP32` | `save_load`, `dynamic_batch`, `dynamic_non_batch` |
| `tensorrt` | `TensorRTBackend` | `TENSORRT_FP32` | `save_load`, `export` (`tensorrt_plan`), `dynamic_batch`, `dynamic_non_batch`, `structured_io`, `multigraph`, `onnx_input` |
| `onnxruntime-cuda` | `ONNXRuntimeBackend` (CUDA EP) | `TENSORRT_FP32` | `save_load`, `export` (`onnx`), `dynamic_batch`, `dynamic_non_batch`, `onnx_input` |
| `onnxruntime-tensorrt` | `ONNXRuntimeBackend` (TensorRT EP) | `TENSORRT_FP32` | `save_load`, `dynamic_batch`, `dynamic_non_batch` |
| `torchao-int8wo` | `TorchAOBackend` (`int8wo`) | `rtol=atol=5e-2` | `save_load` |
| `inductor-jit` | `TorchInductorJitBackend` | default | `save_load`, `jit` |
| `torch-tensorrt-jit` | `TorchTensorRTJitBackend` | `TENSORRT_FP32` | `save_load`, `jit` |

TensorRT-based configs run fp32 without TF32; `tensorrt` also runs without CUDA graphs. `tensorrt_backend(**overrides)` builds
TensorRT variants; `test_precision.py` uses it for fp16, int8, fp8, nvfp4, and CUDA graphs.

To add a backend, add one `BACKENDS` entry. Give it only the capabilities that pass on CI.

## Capabilities

A capability is a feature that a backend supports and is tested on. `with_backends(*capabilities)` runs the test
for each backend that has all of them. It raises `ValueError` when no backend matches.
Before selecting capabilities for a test, read the actual `Capability` names and each backend's `capabilities`
in `tests/functional_ci/backends.py`. Treat that code as the source of truth if the summary below differs.

| Capability | Meaning | Tests |
| --- | --- | --- |
| `save_load` | Tune, `ait.save`, `ait.load` into other weights, same output | `test_backend_lifecycle`, `test_pipeline_lifecycle` |
| `export` | `module.artifact()` runs in its native runtime | `test_exported_artifact` |
| `dynamic_batch` | Batch range from recorded sizes or a user `BatchDim` | `test_recorded_batch_range`, `test_explicit_batch_dim` |
| `dynamic_non_batch` | User `DynamicDim` on non-batch (spatial) dims | `test_explicit_spatial_dims` |
| `structured_io` | Nested dict input, non-tensor args, nested outputs | `test_structured_io` |
| `multigraph` | One wrapped module tunes one graph per input layout | `test_partial_wrap_multigraph` |
| `onnx_input` | Tune an existing ONNX file (inline or external data) | `test_existing_onnx` |
| `jit` | JIT tuning (`jit_config`, `prepare_for_jit_tuning`) | `test_jit_tuning`, `test_pipeline_jit` |

To add a capability, add the name to `Capability`, add it to the backends that pass, and use it in `with_backends`.

### Known failures

Put a strict mark on one backend case through `marks=`:

```python
@with_backends(
    "dynamic_non_batch",
    marks={"torch-tensorrt-aot": pytest.mark.xfail(TORCH_2_8_TO_2_10, reason="... (aitune#118)", strict=True)},
)
```

`marks` raises `ValueError` for a backend id that the capabilities do not select. Do not call `pytest.xfail()` in a
test body, and do not use string conditions on params. Use public issue references such as `aitune#118`, not
internal bug IDs.

## Feature matrix and IDs

A cell ID has the form `<table>.<row>/<column>`. `@pytest.mark.matrix(*templates)` lists the cells that a test
proves. `conftest.py` formats each template with `str.format`:

- `{backend}` is the backend id of the `case` param.
- Other names are pytest params, for example `{case_id}`, `{strategy_id}`, `{mode.value}`.
- A template without `/` gets `/{backend}` added, so it needs a `case` param from `with_backends`.

Cell state is the worst outcome of its tests: `failed` > `skipped` > `passed`. A strict `xfail` that fails counts as
`skipped`; a strict XPASS counts as `failed`. Cell IDs are not validated against the matrix; only the template
formatting is checked.

| Table | Row examples | Column |
| --- | --- | --- |
| `models` | `tensor-inputs`, `static-shapes`, `single-graph`, `dynamic-batch-recorded`, `dynamic-batch-user`, `dynamic-non-batch`, `non-tensor-inputs`, `nested-inputs`, `structured-outputs`, `multiple-graphs`, `pipelines` | backend id or `kernel-selector` |
| `modes` | `aot-tune`, `aot-save-load`, `aot-inspect-wrap`, `aot-export`, `aot-existing-onnx`, `jit-tune_eager`, `jit-tune_deferred`, `jit-inspect` | backend id or `kernel-selector` |
| `strategies` | `one-backend`, `first-wins`, `max-throughput`, `min-latency`, `latency-budget`, `validation-diagnostic`, `find-max-batch-size`, `dry-run` | `aot-tune`, `jit-tune`, `selection-checked` |
| `topology` | backend id or `kernel-selector` | `single-gpu` |
| `other` | `autocast-fp16`, `onnx-int8`, `modelopt-int8`, `modelopt-fp8`, `modelopt-nvfp4`, `cuda-graphs-fp32` | `functional-ci` |

Examples:

```python
@pytest.mark.matrix("modes.aot-export")                  # modes.aot-export/tensorrt, ...
@pytest.mark.matrix("topology.{backend}/single-gpu")     # topology.inductor-aot/single-gpu, ...
@pytest.mark.matrix("modes.jit-{mode.value}")            # modes.jit-tune_eager/inductor-jit, ...
@pytest.mark.matrix("models.pipelines/kernel-selector")  # fixed cell, no backend param
```

With `AITUNE_MATRIX_RESULTS=<path>`, the session writes:

```json
{"run": {"date": "...", "source": "<run URL or SHA>"},
 "cells": {"modes.aot-export/tensorrt": {"state": "passed", "run": "passed: 1 test(s)"}}}
```

For a `skipped` or `failed` cell, `run` is `<state>: <test id> (<reason>)`. CI uploads the JSON with the JUnit XML.
