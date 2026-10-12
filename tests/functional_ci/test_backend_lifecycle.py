# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend lifecycle cases on a toy MLP."""

from collections.abc import Callable
from functools import partial

import pytest
import torch

from aitune.torch.backend import (
    Backend,
    TensorRTBackend,
    TensorRTBackendConfig,
    TorchEagerBackend,
    TorchInductorAotBackend,
    TorchTensorRTAotBackend,
    TorchTensorRTAotBackendConfig,
    TorchTensorRTConfig,
)

from .lifecycle import Tolerance, run_lifecycle
from .models import make_input, make_mlp

# TensorRT fp32 on the SM120 PR runner differs from eager by about 3e-4 even with TF32 disabled.
TENSORRT_FP32 = Tolerance(rtol=1e-3, atol=1e-3)


def _torch_tensorrt_aot() -> Backend:
    """Build a Torch-TensorRT AOT backend that converts the whole graph in fp32."""
    compile_config = TorchTensorRTConfig(enabled_precisions={torch.float32}, min_block_size=1, disable_tf32=True)
    return TorchTensorRTAotBackend(TorchTensorRTAotBackendConfig(compile_config=compile_config))


def _tensorrt() -> Backend:
    """Build a standalone TensorRT backend without TF32 and CUDA graphs."""
    return TensorRTBackend(TensorRTBackendConfig(enable_tf32=False, use_cuda_graphs=False))


BACKENDS: dict[str, tuple[Callable[[], Backend], Tolerance]] = {
    "torch-eager": (TorchEagerBackend, Tolerance()),
    "inductor-aot": (TorchInductorAotBackend, Tolerance()),
    "torch-tensorrt-aot": (_torch_tensorrt_aot, TENSORRT_FP32),
    "tensorrt": (_tensorrt, TENSORRT_FP32),
}


@pytest.mark.parametrize("backend_id", BACKENDS)
def test_backend_lifecycle(backend_id: str, device: torch.device, tmp_path):
    """Tune, infer, save, and reload the toy MLP with one backend."""
    backend_factory, tolerance = BACKENDS[backend_id]
    run_lifecycle(
        backend_factory(),
        partial(make_mlp, device),
        make_input(device),
        tmp_path / "model.ait",
        tolerance,
    )
