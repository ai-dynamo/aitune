# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific
"""Test for device utilities."""

import pytest
import torch

from aitune.torch.utils.device import get_device


def test_get_device_cpu_string():
    """Test get_device with CPU string input."""
    result = get_device("cpu")
    assert isinstance(result, torch.device)
    assert result.type == "cpu"
    assert result.index is None


def test_get_device_cpu_torch_device():
    """Test get_device with CPU torch.device input."""
    cpu_device = torch.device("cpu")
    result = get_device(cpu_device)
    assert isinstance(result, torch.device)
    assert result.type == "cpu"
    assert result.index is None


def test_get_device_cuda_string_no_index():
    """Test get_device with 'cuda' string (should default to cuda:0)."""
    result = get_device("cuda")
    assert isinstance(result, torch.device)
    assert result.type == "cuda"
    assert result.index == 0


def test_get_device_cuda_string_with_index():
    """Test get_device with 'cuda:N' string."""
    for device_id in [0, 1, 2, 10]:
        result = get_device(f"cuda:{device_id}")
        assert isinstance(result, torch.device)
        assert result.type == "cuda"
        assert result.index == device_id


def test_get_device_cuda_torch_device_no_index():
    """Test get_device with cuda torch.device without index."""
    cuda_device = torch.device("cuda")
    result = get_device(cuda_device)
    assert isinstance(result, torch.device)
    assert result.type == "cuda"
    assert result.index == 0


def test_get_device_cuda_torch_device_with_index():
    """Test get_device with cuda torch.device with index."""
    for device_id in [0, 1, 2, 5]:
        cuda_device = torch.device(f"cuda:{device_id}")
        result = get_device(cuda_device)
        assert isinstance(result, torch.device)
        assert result.type == "cuda"
        assert result.index == device_id


def test_get_device_invalid_string_inputs():
    """Test get_device with various invalid string inputs."""
    invalid_inputs = [
        "",
        "gpu",
        "cuda:",
        "cuda:a",
        "cuda:-1",
        "cuda:1.5",
        "cuda: 0",
        " cuda:0",
        "cuda:0 ",
        "CUDA:0",
        "mps",
        "tpu",
        "invalid",
        "cuda:0:0",
    ]

    for invalid_input in invalid_inputs:
        with pytest.raises(ValueError, match="Invalid device"):
            get_device(invalid_input)


def test_get_device_invalid_torch_device_inputs():
    """Test get_device with invalid torch.device inputs."""
    # Test with unsupported device types
    if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        with pytest.raises(ValueError, match="Invalid device"):
            get_device(mps_device)

    # We can't easily create torch.device with invalid types without causing errors,
    # so we'll test with a mock or patch if needed


def test_get_device_none_input():
    """Test get_device with None input."""
    with pytest.raises(ValueError, match="Invalid device"):
        get_device(None)


def test_get_device_integer_input():
    """Test get_device with integer input."""
    with pytest.raises(ValueError, match="Invalid device"):
        get_device(0)


def test_get_device_list_input():
    """Test get_device with list input."""
    with pytest.raises(ValueError, match="Invalid device"):
        get_device(["cuda:0"])


def test_get_device_return_type_consistency():
    """Test that get_device always returns torch.device objects."""
    test_cases = [
        "cpu",
        "cuda",
        "cuda:0",
        "cuda:1",
        torch.device("cpu"),
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    ]

    for test_case in test_cases:
        result = get_device(test_case)
        assert isinstance(result, torch.device)


def test_get_device_large_device_indices():
    """Test get_device with large device indices."""
    large_indices = [99, 100, 127]

    for index in large_indices:
        result = get_device(f"cuda:{index}")
        assert isinstance(result, torch.device)
        assert result.type == "cuda"
        assert result.index == index


def test_get_device_device_indices_from_out_of_range():
    """Test get_device with device indices from out of range."""
    large_indices = [128, 512, 999]

    for index in large_indices:
        with pytest.raises(ValueError, match=f"Invalid device index: {index}. Expected 0-127"):
            get_device(f"cuda:{index}")


def test_get_device_edge_case_zero_index():
    """Test get_device specifically with zero index."""
    # Test string input
    result_str = get_device("cuda:0")
    assert result_str.type == "cuda"
    assert result_str.index == 0

    # Test torch.device input
    result_device = get_device(torch.device("cuda:0"))
    assert result_device.type == "cuda"
    assert result_device.index == 0

    # Test that "cuda" becomes "cuda:0"
    result_default = get_device("cuda")
    assert result_default.type == "cuda"
    assert result_default.index == 0
