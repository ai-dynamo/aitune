# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend lifecycle cases on a toy MLP."""

from collections.abc import Callable
from functools import partial

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
    TorchTensorRTAotBackend,
    TorchTensorRTAotBackendConfig,
    TorchTensorRTConfig,
)

from .artifacts import run_exported_artifact
from .lifecycle import TENSORRT_FP32, Tolerance, run_lifecycle
from .models import make_input, make_mlp


def _torch_tensorrt_aot() -> Backend:
    """Build a Torch-TensorRT AOT backend that converts the whole graph in fp32."""
    compile_config = TorchTensorRTConfig(
        enabled_precisions={torch.float32},
        min_block_size=1,
        disable_tf32=True,
        require_full_compilation=True,
    )
    return TorchTensorRTAotBackend(TorchTensorRTAotBackendConfig(compile_config=compile_config))


def _tensorrt() -> Backend:
    """Build a standalone TensorRT backend without TF32 and CUDA graphs."""
    return TensorRTBackend(TensorRTBackendConfig(enable_tf32=False, use_cuda_graphs=False))


def _onnx_runtime(provider: ONNXExecutionProvider) -> Callable[[], Backend]:
    """Return a factory for an ONNX Runtime backend on one execution provider."""
    return lambda: ONNXRuntimeBackend(ONNXRuntimeBackendConfig(execution_provider=provider))


def _torchao_int8_weight_only() -> Backend:
    """Build a TorchAO int8 weight-only backend compiled without autotuning."""
    return TorchAOBackend(TorchAOBackendConfig(quantization="int8wo", mode="default"))


DEPLOYABLE = {"inductor-aot": "pt2", "tensorrt": "tensorrt_plan", "onnxruntime-cuda": "onnx"}

BACKENDS: dict[str, tuple[Callable[[], Backend], Tolerance]] = {
    "torch-eager": (TorchEagerBackend, Tolerance()),
    "inductor-aot": (TorchInductorAotBackend, Tolerance()),
    "torch-tensorrt-aot": (_torch_tensorrt_aot, TENSORRT_FP32),
    "tensorrt": (_tensorrt, TENSORRT_FP32),
    "onnxruntime-cuda": (_onnx_runtime(ONNXExecutionProvider.CUDA), TENSORRT_FP32),
    "onnxruntime-tensorrt": (_onnx_runtime(ONNXExecutionProvider.TENSORRT), TENSORRT_FP32),
    "torchao-int8wo": (_torchao_int8_weight_only, Tolerance(rtol=5e-2, atol=5e-2)),
}


@pytest.mark.parametrize("backend_id", BACKENDS)
def test_backend_lifecycle(backend_id: str, device: torch.device, tmp_path):
    """Tune, infer, save, and reload the toy MLP; run the exported artifact natively where deployable."""
    backend_factory, tolerance = BACKENDS[backend_id]
    value = make_input(device)
    result = run_lifecycle(backend_factory(), partial(make_mlp, device), value, tmp_path / "model.ait", tolerance)
    if backend_id not in DEPLOYABLE:
        return

    artifact = result.restored.artifact()
    assert artifact.model.format == DEPLOYABLE[backend_id]
    (output,) = run_exported_artifact(artifact, tmp_path / "exported", [value])
    with torch.no_grad():
        expected = make_mlp(device)(value)
    torch.testing.assert_close(output, expected, rtol=tolerance.rtol, atol=tolerance.atol)
