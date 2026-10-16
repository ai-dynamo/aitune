# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend table for cross-backend cases: factory, tolerance, and the features each backend is tested on."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import pytest
import torch

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

Capability = Literal["save_load", "dynamic_batch", "structured_io", "multigraph", "onnx_input", "jit"]


@dataclass(frozen=True)
class BackendCase:
    """One backend in the table; `artifact_format` is set when the backend exports a deployable artifact."""

    factory: Callable[[], Backend]
    tolerance: Tolerance
    capabilities: frozenset[Capability]
    artifact_format: str | None = None


def _torch_tensorrt_config() -> TorchTensorRTConfig:
    """Convert the whole graph in fp32 without TF32."""
    return TorchTensorRTConfig(
        enabled_precisions={torch.float32}, min_block_size=1, disable_tf32=True, require_full_compilation=True
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


def _tensorrt() -> Backend:
    """Build a standalone TensorRT backend without TF32 and CUDA graphs."""
    return TensorRTBackend(TensorRTBackendConfig(enable_tf32=False, use_cuda_graphs=False))


def _onnx_runtime(provider: ONNXExecutionProvider) -> Callable[[], Backend]:
    """Return a factory for an ONNX Runtime backend on one execution provider."""
    return lambda: ONNXRuntimeBackend(ONNXRuntimeBackendConfig(execution_provider=provider))


def _torchao_int8_weight_only() -> Backend:
    """Build a TorchAO int8 weight-only backend compiled without autotuning."""
    return TorchAOBackend(TorchAOBackendConfig(quantization="int8wo", mode="default"))


BACKENDS: dict[str, BackendCase] = {
    "torch-eager": BackendCase(TorchEagerBackend, Tolerance(), frozenset({"save_load"})),
    "inductor-aot": BackendCase(
        TorchInductorAotBackend,
        Tolerance(),
        frozenset({"save_load", "dynamic_batch", "structured_io", "multigraph"}),
        artifact_format="pt2",
    ),
    "torch-tensorrt-aot": BackendCase(_torch_tensorrt_aot, TENSORRT_FP32, frozenset({"save_load"})),
    "tensorrt": BackendCase(
        _tensorrt,
        TENSORRT_FP32,
        frozenset({"save_load", "dynamic_batch", "structured_io", "multigraph", "onnx_input"}),
        artifact_format="tensorrt_plan",
    ),
    "onnxruntime-cuda": BackendCase(
        _onnx_runtime(ONNXExecutionProvider.CUDA),
        TENSORRT_FP32,
        frozenset({"save_load", "dynamic_batch", "onnx_input"}),
        artifact_format="onnx",
    ),
    "onnxruntime-tensorrt": BackendCase(
        _onnx_runtime(ONNXExecutionProvider.TENSORRT), TENSORRT_FP32, frozenset({"save_load"})
    ),
    "torchao-int8wo": BackendCase(_torchao_int8_weight_only, Tolerance(rtol=5e-2, atol=5e-2), frozenset({"save_load"})),
    "inductor-jit": BackendCase(TorchInductorJitBackend, Tolerance(), frozenset({"jit"})),
    "torch-tensorrt-jit": BackendCase(_torch_tensorrt_jit, TENSORRT_FP32, frozenset({"jit"})),
}


def with_backends(*capabilities: Capability) -> pytest.MarkDecorator:
    """Parametrize `case` over every backend that has all `capabilities`; other backends get no case."""
    ids = [backend_id for backend_id, case in BACKENDS.items() if case.capabilities.issuperset(capabilities)]
    return pytest.mark.parametrize("case", [BACKENDS[backend_id] for backend_id in ids], ids=ids)
