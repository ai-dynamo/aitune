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

from aitune.torch.inspecting.module_inspector import ModuleInspector


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))


@pytest.fixture
def nested_model():
    """Create a nested model for testing."""

    class NestedModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(20, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.linear(x))

    return nn.Sequential(nn.Linear(10, 20), NestedModule(), nn.Linear(10, 5))


@pytest.fixture
def inspector():
    """Create a ModuleInspector instance."""
    return ModuleInspector()


def test_initialization(inspector):
    """Test ModuleInspector initialization."""
    assert inspector._module_info == {}
    assert inspector._original_forward == {}
    assert inspector._inspected_objects == set()
    assert inspector._max_recursion_depth == 5


def test_reset(inspector, simple_model):
    """Test reset functionality."""
    # First inspect a model
    inspector.inspect(simple_model)
    assert len(inspector._module_info) > 0
    assert len(inspector._original_forward) > 0
    assert len(inspector._inspected_objects) > 0

    # Then reset
    inspector.reset()
    assert inspector._module_info == {}
    assert inspector._original_forward == {}
    assert inspector._inspected_objects == set()


def test_inspect_simple_model(inspector, simple_model):
    """Test inspecting a simple model."""
    inspector.inspect(simple_model)

    # Check that all modules were registered
    assert len(inspector._module_info) == 4  # Sequential, Linear, ReLU, Linear

    # Check module hierarchy
    root_module = next(iter(inspector._module_info.values()))
    assert root_module.name == "Sequential"
    assert len(root_module.children) == 3


def test_inspect_nested_model(inspector, nested_model):
    """Test inspecting a nested model."""
    inspector.inspect(nested_model)

    # Check that all modules were registered
    assert len(inspector._module_info) == 6  # Sequential, Linear, NestedModule, Linear, Linear, ReLU

    # Check module hierarchy
    root_module = next(iter(inspector._module_info.values()))
    assert root_module.name == "Sequential"
    assert len(root_module.children) == 3


def test_get_modules(inspector, simple_model):
    """Test getting executed modules."""
    inspector.inspect(simple_model)

    # Initially no modules should be executed
    assert len(inspector.get_modules()) == 0

    # Execute the model
    input_tensor = torch.randn(1, 10)
    simple_model(input_tensor)

    # Now we should have executed modules
    executed_modules = inspector.get_modules()
    assert len(executed_modules) > 0
    assert all(module.forward_called for module in executed_modules)


def test_wrap_forward_methods(inspector, simple_model):
    """Test wrapping forward methods."""
    inspector.inspect(simple_model)

    # Check that forward methods were wrapped
    assert len(inspector._original_forward) == 4  # One for each module

    # Check that the wrapped forward method tracks execution
    input_tensor = torch.randn(1, 10)
    simple_model(input_tensor)

    for module_info in inspector._module_info.values():
        assert module_info.forward_called
        assert module_info.execution_count == 1
        assert module_info.total_execution_time > 0


def test_inspect_builtin_object(inspector):
    """Test inspecting a built-in object."""
    builtin_obj = [1, 2, 3]
    inspector.inspect(builtin_obj)

    # Built-in objects should be skipped
    assert len(inspector._module_info) == 0
    assert len(inspector._original_forward) == 0


def test_inspect_module_with_custom_forward(inspector):
    """Test inspecting a module with a custom forward method."""

    class CustomModule(nn.Module):
        def forward(self, x, extra_arg=None):
            return x + 1 if extra_arg else x

    model = CustomModule()
    inspector.inspect(model)

    # Execute with different arguments
    input_tensor = torch.randn(1, 10)
    model(input_tensor)
    model(input_tensor, extra_arg=True)

    # Check execution tracking
    module_info = next(iter(inspector._module_info.values()))
    assert module_info.execution_count == 2
    assert len(module_info.output_types) == 2


def test_inspect_module_with_errors(inspector):
    """Test inspecting a module that raises errors."""

    class ErrorModule(nn.Module):
        def forward(self, x):
            raise RuntimeError("Test error")

    model = ErrorModule()
    inspector.inspect(model)

    # The inspection should complete without errors
    assert len(inspector._module_info) == 1

    # The forward call should raise the error
    with pytest.raises(RuntimeError):
        model(torch.randn(1, 10))


def test_inspect_module_with_cuda(inspector):
    """Test inspecting a module on CUDA if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model = nn.Linear(10, 20).cuda()
    inspector.inspect(model)

    # Execute on CUDA
    input_tensor = torch.randn(1, 10).cuda()
    model(input_tensor)

    # Check execution tracking
    module_info = next(iter(inspector._module_info.values()))
    assert module_info.forward_called
    assert module_info.execution_count == 1
    assert module_info.total_execution_time > 0


def test_inspect_module_with_multiple_forward_calls(inspector, simple_model):
    """Test inspecting a module with multiple forward calls."""
    inspector.inspect(simple_model)

    # Execute multiple times
    input_tensor = torch.randn(1, 10)
    for _ in range(3):
        simple_model(input_tensor)

    # Check execution tracking
    for module_info in inspector._module_info.values():
        assert module_info.forward_called
        assert module_info.execution_count == 3
        assert module_info.total_execution_time > 0
        assert module_info.average_execution_time > 0


def test_inspect_module_with_complex_output(inspector):
    """Test inspecting a module with complex output types."""

    class ComplexOutputModule(nn.Module):
        def forward(self, x):
            return {"output1": x + 1, "output2": (x * 2, x * 3), "output3": [x, x + 1]}

    model = ComplexOutputModule()
    inspector.inspect(model)

    # Execute
    input_tensor = torch.randn(1, 10)
    model(input_tensor)

    # Check output type tracking
    module_info = next(iter(inspector._module_info.values()))
    assert len(module_info.output_types) == 1
    output_info = module_info.output_types[0]
    assert output_info["type"] == "dict"
