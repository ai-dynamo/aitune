# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reduced-precision, quantized, and CUDA-graph TensorRT lifecycles, gated on GPU capability."""

from functools import partial

import pytest
import torch
from packaging.version import Version

from aitune.torch.backend import ONNXAutoCastConfig, ONNXQuantizationConfig, TorchQuantizationConfig
from tests.functional_ci.backends import tensorrt_backend
from tests.functional_ci.lifecycle import TENSORRT_FP32, Tolerance, assert_matches_eager, run_lifecycle
from tests.functional_ci.models import make_input, make_mlp

FP16 = Tolerance(rtol=1e-2, atol=1e-2)
QUANTIZED = Tolerance(rtol=1e-1, atol=1e-1)

# id: (`tensorrt_backend` overrides, tolerance, minimum compute capability)
PRECISION_CASES = {
    "autocast-fp16": ({"quantization_config": ONNXAutoCastConfig(precision="fp16")}, FP16, (7, 0)),
    "onnx-int8": ({"quantization_config": ONNXQuantizationConfig(precision="int8")}, QUANTIZED, (7, 5)),
    "modelopt-int8": ({"quantization_config": TorchQuantizationConfig("INT8_DEFAULT_CFG")}, QUANTIZED, (7, 5)),
    "modelopt-fp8": ({"quantization_config": TorchQuantizationConfig("FP8_DEFAULT_CFG")}, QUANTIZED, (8, 9)),
    "modelopt-nvfp4": ({"quantization_config": TorchQuantizationConfig("NVFP4_DEFAULT_CFG")}, QUANTIZED, (10, 0)),
    "cuda-graphs-fp32": ({"use_cuda_graphs": True}, TENSORRT_FP32, (7, 0)),
}

TORCH_2_15_OR_NEWER = Version(torch.__version__).release[:2] >= (2, 15)
MODELOPT_TORCH_EXPORT_XFAIL = pytest.mark.xfail(
    TORCH_2_15_OR_NEWER,
    reason="torch.export of a ModelOpt-quantized module fails on torch>=2.15: fake tensor in the exported program "
    "constants (input_quantizer.lifted_tensor_0); no fix in nvidia-modelopt<=0.49.0rc0",
    raises=AssertionError,
    strict=True,
)


@pytest.mark.matrix("other.{case_id}/functional-ci")
@pytest.mark.parametrize(
    "case_id",
    [
        pytest.param(case_id, marks=MODELOPT_TORCH_EXPORT_XFAIL if case_id.startswith("modelopt-") else ())
        for case_id in PRECISION_CASES
    ],
)
def test_tensorrt_precision_lifecycle(case_id: str, device: torch.device, tmp_path):
    """Tune, save, and reload a TensorRT engine with the requested precision or runtime option, then replay it."""
    config_kwargs, tolerance, min_capability = PRECISION_CASES[case_id]
    capability = torch.cuda.get_device_capability(device)
    if capability < min_capability:
        pytest.skip(f"{case_id} needs compute capability {min_capability}, GPU has {capability}")

    backend = tensorrt_backend(**config_kwargs)
    restored = run_lifecycle(backend, partial(make_mlp, device), make_input(device), tmp_path / "model.ait", tolerance)
    # Two new inputs check that a replayed engine (for example a CUDA graph) does not reuse stale input buffers.
    assert_matches_eager(restored, make_mlp(device), device, tolerance, seeds=(11, 12))
