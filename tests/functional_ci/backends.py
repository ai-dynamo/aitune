# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend table for cross-backend cases: factory, tolerance, and the features each backend is tested on."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import pytest

from aitune.torch.backend import (
    Backend,
    ONNXExecutionProvider,
    ONNXRuntimeBackend,
    ONNXRuntimeBackendConfig,
    TensorRTBackend,
    TensorRTBackendConfig,
    TorchAOBackend,
    TorchAOBackendConfig,
    TorchEagerBackend,
    TorchInductorAotBackend,
    TorchInductorJitBackend,
    TorchTensorRTAotBackend,
    TorchTensorRTAotBackendConfig,
    TorchTensorRTConfig,
    TorchTensorRTJitBackend,
    TorchTensorRTJitBackendConfig,
)
from tests.functional_ci.lifecycle import TENSORRT_FP32, Tolerance

Tag = Literal[
    "base",
    "save_load",
    "export",
    "dynamic_batch",
    "dynamic_non_batch",
    "structured_io",
    "multigraph",
    "onnx_input",
    "jit",
    "graph_breaks",
    "torch_export",
    "onnx_export",
]
Capability = Tag


@dataclass(frozen=True)
class BackendCase:
    """One backend in the table; `artifact_format` is the deployable artifact format of `export` backends."""

    factory: Callable[[], Backend]
    tolerance: Tolerance
    tags: frozenset[Tag]
    artifact_format: str | None = None

    def __post_init__(self):
        """Keep the `export` tags and `artifact_format` in step."""
        assert ("export" in self.tags) == (self.artifact_format is not None), "set both or neither"


def _torch_tensorrt_config() -> TorchTensorRTConfig:
    """Convert the whole graph in fp32 without TF32; fail instead of falling back to PyTorch."""
    return TorchTensorRTConfig(
        min_block_size=1,
        disable_tf32=True,
        require_full_compilation=True,
        pass_through_build_failures=True,
    )


def _torch_tensorrt_aot() -> Backend:
    """Build a Torch-TensorRT AOT backend."""
    return TorchTensorRTAotBackend(TorchTensorRTAotBackendConfig(compile_config=_torch_tensorrt_config()))


def _torch_tensorrt_jit() -> Backend:
    """Build a Torch-TensorRT JIT backend."""
    # Dynamic batch compilation fails the max-shape correctness check (sympy `Infinity` bound), so compile per shape.
    return TorchTensorRTJitBackend(
        TorchTensorRTJitBackendConfig(compile_config=_torch_tensorrt_config(), dynamic=False)
    )


def tensorrt_backend(**overrides) -> Backend:
    """Build a standalone TensorRT backend without TF32 and CUDA graphs; `overrides` replace config fields."""
    config: dict[str, Any] = {"enable_tf32": False, "use_cuda_graphs": False, **overrides}
    return TensorRTBackend(TensorRTBackendConfig(**config))


def _onnx_runtime(provider: ONNXExecutionProvider) -> Callable[[], Backend]:
    """Return a factory for an ONNX Runtime backend on one execution provider."""
    return lambda: ONNXRuntimeBackend(ONNXRuntimeBackendConfig(execution_provider=provider))


def _torchao_int8_weight_only() -> Backend:
    """Build a TorchAO int8 weight-only backend compiled without autotuning."""
    return TorchAOBackend(TorchAOBackendConfig(quantization="int8wo", mode="default"))


def _tags(*tags: Tag) -> frozenset[Tag]:
    """Build a tags set; the `Tag` parameter lets linters reject a misspelled name."""
    return frozenset(tags)


BACKENDS: dict[str, BackendCase] = {
    "torch-eager": BackendCase(
        TorchEagerBackend,
        Tolerance(),
        _tags(
            "base",
            "save_load",
            "graph_breaks",
        ),
    ),
    "inductor-aot": BackendCase(
        TorchInductorAotBackend,
        Tolerance(),
        _tags(
            "base",
            "save_load",
            "export",
            "dynamic_batch",
            "dynamic_non_batch",
            "structured_io",
            "multigraph",
            "torch_export",
        ),
        artifact_format="pt2",
    ),
    "torch-tensorrt-aot": BackendCase(
        _torch_tensorrt_aot,
        TENSORRT_FP32,
        _tags(
            "base",
            "save_load",
            "dynamic_batch",
            "dynamic_non_batch",
            "structured_io",
            "torch_export",
        ),
    ),
    "tensorrt": BackendCase(
        tensorrt_backend,
        TENSORRT_FP32,
        _tags(
            "base",
            "save_load",
            "export",
            "dynamic_batch",
            "dynamic_non_batch",
            "structured_io",
            "multigraph",
            "onnx_input",
            "onnx_export",
        ),
        artifact_format="tensorrt_plan",
    ),
    "onnxruntime-cuda": BackendCase(
        _onnx_runtime(ONNXExecutionProvider.CUDA),
        TENSORRT_FP32,
        _tags(
            "base",
            "save_load",
            "export",
            "dynamic_batch",
            "dynamic_non_batch",
            "onnx_input",
            "onnx_export",
        ),
        artifact_format="onnx",
    ),
    "onnxruntime-tensorrt": BackendCase(
        _onnx_runtime(ONNXExecutionProvider.TENSORRT),
        TENSORRT_FP32,
        _tags(
            "base",
            "save_load",
            "dynamic_batch",
            "dynamic_non_batch",
            "onnx_export",
        ),
    ),
    "torchao-int8wo": BackendCase(
        _torchao_int8_weight_only,
        Tolerance(rtol=5e-2, atol=5e-2),
        _tags(
            "base",
            "save_load",
            "graph_breaks",
        ),
    ),
    "inductor-jit": BackendCase(
        TorchInductorJitBackend,
        Tolerance(),
        _tags(
            "base",
            "save_load",
            "jit",
        ),
    ),
    "torch-tensorrt-jit": BackendCase(
        _torch_tensorrt_jit,
        TENSORRT_FP32,
        _tags(
            "base",
            "save_load",
            "jit",
        ),
    ),
}


def with_backends(*tags: Tag, marks: Mapping[str, pytest.MarkDecorator] | None = None) -> pytest.MarkDecorator:
    """Parametrize `case` over backends with all `tags`; raise if none match.

    `marks` maps a backend id to marks for that case only, e.g. a strict `pytest.mark.xfail` for a known failure.
    """
    marks = marks or {}
    selected = {backend_id: case for backend_id, case in BACKENDS.items() if case.tags.issuperset(tags)}
    if not selected:
        raise ValueError(f"no backend in BACKENDS has all tags {tags}")
    if unknown := marks.keys() - selected.keys():
        raise ValueError(f"marks for backends without tags {tags}: {sorted(unknown)}")
    return pytest.mark.parametrize(
        "case",
        [pytest.param(case, marks=marks.get(backend_id, ()), id=backend_id) for backend_id, case in selected.items()],
    )
