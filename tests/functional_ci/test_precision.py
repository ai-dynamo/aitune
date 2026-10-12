# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reduced-precision, quantized, and CUDA-graph TensorRT lifecycles, gated on GPU capability."""

from functools import partial

import pytest
import torch

from aitune.torch.backend import (
    ONNXAutoCastConfig,
    ONNXQuantizationConfig,
    TensorRTBackend,
    TensorRTBackendConfig,
    TorchQuantizationConfig,
)

from .lifecycle import TENSORRT_FP32, Tolerance, run_lifecycle
from .models import make_input, make_mlp

FP16 = Tolerance(rtol=1e-2, atol=1e-2)
QUANTIZED = Tolerance(rtol=1e-1, atol=1e-1)

# id: (TensorRTBackendConfig kwargs, tolerance, minimum compute capability)
PRECISION_CASES = {
    "autocast-fp16": ({"quantization_config": ONNXAutoCastConfig(precision="fp16")}, FP16, (7, 0)),
    "onnx-int8": ({"quantization_config": ONNXQuantizationConfig(precision="int8")}, QUANTIZED, (7, 5)),
    "modelopt-int8": ({"quantization_config": TorchQuantizationConfig("INT8_DEFAULT_CFG")}, QUANTIZED, (7, 5)),
    "modelopt-fp8": ({"quantization_config": TorchQuantizationConfig("FP8_DEFAULT_CFG")}, QUANTIZED, (8, 9)),
    "modelopt-nvfp4": ({"quantization_config": TorchQuantizationConfig("NVFP4_DEFAULT_CFG")}, QUANTIZED, (10, 0)),
    "cuda-graphs-fp32": ({"use_cuda_graphs": True}, TENSORRT_FP32, (7, 0)),
}


@pytest.mark.parametrize("case_id", PRECISION_CASES)
def test_tensorrt_precision_lifecycle(case_id: str, device: torch.device, tmp_path):
    """Tune, infer, save, and reload a TensorRT engine built with the requested precision or runtime option."""
    config_kwargs, tolerance, min_capability = PRECISION_CASES[case_id]
    capability = torch.cuda.get_device_capability(device)
    if capability < min_capability:
        pytest.skip(f"{case_id} needs compute capability {min_capability}, GPU has {capability}")

    backend = TensorRTBackend(TensorRTBackendConfig(enable_tf32=False, **{"use_cuda_graphs": False, **config_kwargs}))
    result = run_lifecycle(backend, partial(make_mlp, device), make_input(device), tmp_path / "model.ait", tolerance)
    if config_kwargs.get("use_cuda_graphs"):
        reference = make_mlp(device)
        with torch.no_grad():
            for seed in (11, 12):
                value = make_input(device, seed=seed)
                torch.testing.assert_close(
                    result.restored(value), reference(value), rtol=tolerance.rtol, atol=tolerance.atol
                )
