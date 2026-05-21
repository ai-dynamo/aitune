# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TorchQuantizer."""

import modelopt.torch.quantization as mtq
import pytest
import torch

from aitune.torch.backend.tensorrt.torch_quantization import TorchQuantizationConfig, TorchQuantizer
from tests.toy_models.torch_models import ToyTorchModel
from tests.utilities.helpers import requires_cuda


def test_create_forward_loop():
    """Test the creation of a forward loop function."""
    quantizer = TorchQuantizer()
    forward_loop = quantizer._create_forward_loop([(torch.randn(1, 3, 224, 224),), (torch.randn(1, 3, 224, 224),)])
    assert callable(forward_loop)
    assert forward_loop(torch.nn.Identity()) is None


@requires_cuda
def test_prepare_calibration_sample():
    """Test the preparation of calibration sample."""
    quantizer = TorchQuantizer()
    cpu_sample = (
        (torch.randn(1, 3, 224, 224, device="cpu"),),  # args
        {},  # kwargs
    )

    calibration_sample = quantizer._prepare_calibration_sample(
        sample=cpu_sample,
        device="cuda",
    )

    # validate returned calibration sample
    assert len(calibration_sample) == len(cpu_sample) == 2

    # validate args
    assert len(calibration_sample[0]) == len(cpu_sample[0]) == 1
    assert calibration_sample[0][0].shape == cpu_sample[0][0].shape == (1, 3, 224, 224)
    assert calibration_sample[0][0].device.type == "cuda" != cpu_sample[0][0].device.type

    # validate kwargs
    assert calibration_sample[1] == cpu_sample[1] == {}


def test_get_quantization_config():
    """Test the retrieval of quantization configuration."""
    quantizer = TorchQuantizer()
    assert quantizer._get_quantization_config("NVFP4_DEFAULT_CFG") == mtq.NVFP4_DEFAULT_CFG
    assert quantizer._get_quantization_config("NVFP4_FP8_MHA_CONFIG") == mtq.NVFP4_FP8_MHA_CONFIG
    assert quantizer._get_quantization_config("FP8_DEFAULT_CFG") == mtq.FP8_DEFAULT_CFG
    assert quantizer._get_quantization_config("INT8_DEFAULT_CFG") == mtq.INT8_DEFAULT_CFG
    assert quantizer._get_quantization_config("INT8_SMOOTHQUANT_CFG") == mtq.INT8_SMOOTHQUANT_CFG
    assert quantizer._get_quantization_config("INT4_AWQ_CFG") == mtq.INT4_AWQ_CFG


def test_get_quantization_config_invalid_config():
    """Test the retrieval of quantization configuration with invalid config."""
    quantizer = TorchQuantizer()
    with pytest.raises(ValueError):
        quantizer._get_quantization_config("INVALID_CFG")


@requires_cuda
def test_quantize():
    """Test the quantization of a model."""
    quantizer = TorchQuantizer()
    model = ToyTorchModel()
    sample = ((model.sample(),), {})
    config = TorchQuantizationConfig()
    quantized_model = quantizer.quantize(model, sample, config)
    assert quantized_model is not None
    assert isinstance(quantized_model, torch.nn.Module)
