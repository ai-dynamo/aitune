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
import pytest
import torch
import torch.nn as nn

from aitune.torch.inspecting.module_info import ModuleInfo


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))


@pytest.fixture
def module_info(simple_model):
    """Create a ModuleInfo instance for testing."""
    return ModuleInfo(
        name="test_module",
        module=simple_model,
        parent=None,
        forward_called=False,
        execution_count=0,
        total_execution_time=0.0,
        output_types=[],
    )


def test_module_type(module_info, simple_model):
    """Test module_type property."""
    assert module_info.module_type is type(simple_model)


def test_average_execution_time(module_info):
    """Test average_execution_time property."""
    # Test when execution_count is 0
    assert module_info.average_execution_time == pytest.approx(0.0)

    # Test with some execution time
    module_info.execution_count = 2
    module_info.total_execution_time = 4.0
    assert module_info.average_execution_time == pytest.approx(2.0)


def test_num_layers(module_info):
    """Test num_layers property."""
    # The simple model has 3 layers (Linear, ReLU, Linear)
    assert module_info.num_layers == 3


def test_num_parameters(module_info):
    """Test num_parameters property."""
    # Calculate expected parameters:
    # First Linear: 10 * 20 + 20 (bias) = 220
    # Second Linear: 20 * 5 + 5 (bias) = 105
    # Total: 325 parameters
    assert module_info.num_parameters == 325


def test_precisions(module_info):
    """Test precisions property."""
    # By default, the model should use float32
    precisions = module_info.precisions
    assert len(precisions) == 1
    assert torch.float32 in precisions


def test_execution_tracking(module_info):
    """Test execution tracking properties."""
    # Initial state
    assert not module_info.forward_called
    assert module_info.execution_count == 0
    assert module_info.total_execution_time == pytest.approx(0.0)

    # Update execution state
    module_info.forward_called = True
    module_info.execution_count = 1
    module_info.total_execution_time = 1.5

    assert module_info.forward_called
    assert module_info.execution_count == 1
    assert module_info.total_execution_time == pytest.approx(1.5)
    assert module_info.average_execution_time == pytest.approx(1.5)


def test_output_types(module_info):
    """Test output_types property."""
    # Initial state
    assert len(module_info.output_types) == 0

    # Add output type information
    output_info = {"type": "Tensor", "shape": [1, 5]}
    module_info.output_types.append(output_info)

    assert len(module_info.output_types) == 1
    assert module_info.output_types[0] == output_info
