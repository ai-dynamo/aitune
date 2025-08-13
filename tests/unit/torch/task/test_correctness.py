# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for correctness checking functionality."""

import numpy as np
import pytest
import torch

from aitune.torch.task.correctness import check_output_correctness


def test_check_output_correctness_valid():
    """Test check_output_correctness with valid outputs."""
    outputs = {"output1": torch.tensor([1.0, 2.0, 3.0]), "output2": np.array([4.0, 5.0, 6.0])}
    check_output_correctness(outputs)


def test_check_output_correctness_nan():
    """Test check_output_correctness with NaN values."""
    outputs = {"output1": torch.tensor([1.0, float("nan"), 3.0]), "output2": np.array([4.0, 5.0, 6.0])}
    with pytest.raises(ValueError, match="contains NaN values"):
        check_output_correctness(outputs)


def test_check_output_correctness_inf():
    """Test check_output_correctness with infinity values."""
    outputs = {"output1": torch.tensor([1.0, 2.0, 3.0]), "output2": np.array([4.0, float("inf"), 6.0])}
    with pytest.raises(ValueError, match="contains infinity values"):
        check_output_correctness(outputs)


def test_check_output_correctness_just_a_string():
    """Test check_output_correctness with just a string."""
    outputs = {
        "output1": "not a tensor",
    }
    check_output_correctness(outputs)


def test_check_output_correctness_list():
    """Test check_output_correctness with a list of tensors and numpy arrays."""
    outputs = [torch.tensor([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    check_output_correctness(outputs)


def test_check_output_correctness_sample():
    """Test check_output_correctness with a sample output."""
    outputs = (
        (torch.tensor([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])),
        {"output1": torch.tensor([1.0, 2.0, 3.0]), "output2": np.array([4.0, 5.0, 6.0])},
    )
    check_output_correctness(outputs)


def test_check_output_correctness_sample_scalar_nan():
    """Test check_output_correctness with a sample output."""
    outputs = (
        (float("nan"), torch.tensor([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])),
        {"output1": torch.tensor([1.0, 2.0, 3.0]), "output2": np.array([4.0, 5.0, 6.0])},
    )
    with pytest.raises(ValueError, match="is not finite"):
        check_output_correctness(outputs)
