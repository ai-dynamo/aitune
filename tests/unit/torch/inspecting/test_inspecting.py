# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn

from aitune.torch.distributed import DistributedOutcome
from aitune.torch.inspecting import inspect
from aitune.torch.inspecting.inspecting import _run_inference, _synchronize_inspection_result
from aitune.torch.inspecting.module_info import InspectedModulesInfo, ModuleInfo

TEST_NUMBER_OF_ITERATIONS = 10


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
            self.linear = nn.Linear(10, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.linear(x))

    return NestedModule()


@pytest.fixture
def custom_object():
    """Create a custom object for testing."""

    class CustomObject:
        def __init__(self):
            self.num_iterations = 4
            self.linear1 = nn.Linear(10, 10)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(10, 10)

        def __call__(self, x):
            # Call linear1 multiple times
            for _ in range(self.num_iterations):
                x = self.linear1(x)
            # Then continue with the rest of the forward pass
            x = self.relu(x)
            x = self.linear2(x)
            return x

    return CustomObject()


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return torch.randn(10, 10)  # 10 samples of size 10


def test_inspect_simple_model(simple_model, sample_dataset):
    """Test inspecting a simple model."""
    # When inspecting the model
    modules_info = inspect(simple_model, sample_dataset)

    # Then verify the results
    assert isinstance(modules_info, InspectedModulesInfo)

    modules = modules_info.get_modules()

    assert len(modules) > 0
    assert all(isinstance(info, ModuleInfo) for info in modules)

    # Check that the model was executed
    assert any(info.forward_called for info in modules)
    assert any(info.execution_count > 0 for info in modules)
    assert any(info.total_execution_time > 0 for info in modules)


def test_inspect_synchronizes_complete_inference_calls(mocker, simple_model, sample_dataset):
    """Test that inspection synchronizes ranks around complete model calls."""
    barrier = mocker.patch("aitune.torch.inspecting.inspecting.coordinator.barrier")
    broadcast = mocker.patch(
        "aitune.torch.inspecting.inspecting.coordinator.broadcast_from_rank0",
        side_effect=lambda value: value,
    )

    modules_info = inspect(simple_model, sample_dataset, number_of_iterations=3, warmup_iterations=2)

    assert barrier.call_count == 2 * (3 + 2)
    modules = modules_info.get_modules()
    broadcast.assert_called_once_with((
        modules_info._total_execution_time,
        tuple(
            (
                module.object_path,
                f"{module.module_type.__module__}.{module.module_type.__qualname__}",
                module.execution_count,
                module.total_execution_time,
            )
            for module in modules
        ),
    ))


def test_inspection_inference_coordinates_failures(mocker):
    error = RuntimeError("inference failed")
    outcome = mocker.patch(
        "aitune.torch.inspecting.inspecting.coordinator.outcome",
        return_value=DistributedOutcome(("RuntimeError",)),
    )
    mocker.patch("aitune.torch.inspecting.inspecting.coordinator.barrier")

    with pytest.raises(RuntimeError, match="inference failed"):
        _run_inference(mocker.Mock(side_effect=error), (), {})

    outcome.assert_called_once_with(error)


def test_inspection_uses_rank_zero_order_and_timings_for_selection(mocker):
    modules = [
        ModuleInfo(module=nn.ReLU(), name="first", object_path="first", execution_count=1, total_execution_time=0.9),
        ModuleInfo(module=nn.ReLU(), name="second", object_path="second", execution_count=1, total_execution_time=0.1),
    ]
    module_type = "torch.nn.modules.activation.ReLU"
    broadcast = mocker.patch(
        "aitune.torch.inspecting.inspecting.coordinator.broadcast_from_rank0",
        return_value=(1.0, (("second", module_type, 1, 0.8), ("first", module_type, 1, 0.2))),
    )

    total_execution_time, synchronized_modules = _synchronize_inspection_result(2.0, modules)
    modules_info = InspectedModulesInfo(total_execution_time, number_of_batches=1)
    for module in synchronized_modules:
        modules_info.add_module(module)

    assert [module.name for module in modules_info.get_modules()] == ["second", "first"]
    assert [module.name for module in modules_info.get_modules(limit=1)] == ["second"]
    assert [module.name for module in modules_info.get_modules(min_execution_ratio=0.5)] == ["second"]
    broadcast.assert_called_once_with((2.0, (("first", module_type, 1, 0.9), ("second", module_type, 1, 0.1))))


def test_inspection_rejects_modules_that_do_not_match_rank_zero(mocker):
    modules = [
        ModuleInfo(module=nn.ReLU(), name="local", object_path="local", execution_count=1, total_execution_time=0.1)
    ]
    mocker.patch(
        "aitune.torch.inspecting.inspecting.coordinator.broadcast_from_rank0",
        return_value=(1.0, (("remote", "torch.nn.modules.activation.ReLU", 1, 0.1),)),
    )

    with pytest.raises(RuntimeError, match="modules differ from rank zero"):
        _synchronize_inspection_result(1.0, modules)


def test_inspect_nested_model(nested_model, sample_dataset):
    """Test inspecting a nested model."""
    # When inspecting the nested model
    modules_info = inspect(nested_model.linear, sample_dataset, number_of_iterations=TEST_NUMBER_OF_ITERATIONS)

    # Then verify the results
    assert isinstance(modules_info, InspectedModulesInfo)

    modules = modules_info.get_modules()

    assert len(modules) > 0

    # Verify execution tracking
    assert modules[0].module == nested_model.linear
    assert modules[0].forward_called is True
    assert modules[0].execution_count == TEST_NUMBER_OF_ITERATIONS


def test_inspect_nested_model_with_inference_function(nested_model, sample_dataset):
    """Test inspecting a nested model with an inference function."""

    def inference_function(x):
        return nested_model.forward(x)

    # When inspecting the nested model
    modules_info = inspect(
        nested_model,
        sample_dataset,
        inference_function=inference_function,
        number_of_iterations=TEST_NUMBER_OF_ITERATIONS,
    )

    # Then verify the results
    assert isinstance(modules_info, InspectedModulesInfo)

    modules = modules_info.get_modules()

    assert len(modules) > 0

    # Verify execution tracking
    assert modules[0].module == nested_model
    assert modules[0].forward_called is True
    assert modules[0].execution_count == TEST_NUMBER_OF_ITERATIONS


def test_inspect_with_dataloader(simple_model):
    """Test inspecting with a DataLoader."""
    from aitune.torch.dataloader import DataLoaderFactory

    # Create a simple dataloader
    dataset = torch.randn(10, 10)
    dataloader = DataLoaderFactory(dataset=dataset)

    # When inspecting with dataloader
    modules_info = inspect(simple_model, dataloader)

    # Then verify the results
    assert isinstance(modules_info, InspectedModulesInfo)

    modules = modules_info.get_modules()

    assert len(modules) > 0
    assert any(info.forward_called for info in modules)


def test_inspect_with_complex_output(sample_dataset):
    """Test inspecting a model with complex output types."""

    class ComplexOutputModel(nn.Module):
        def forward(self, x):
            return {"output1": x + 1, "output2": (x * 2, x * 3), "output3": [x, x + 1]}

    model = ComplexOutputModel()

    # When inspecting the model
    modules_info = inspect(model, sample_dataset)

    # Then verify the results
    assert isinstance(modules_info, InspectedModulesInfo)

    modules = modules_info.get_modules()

    assert len(modules) > 0

    # Check output type tracking
    module_info = next(iter(modules))
    assert len(module_info.output_types) > 0
    assert module_info.output_types[0]["type"] == "dict"


def test_inspect_with_cuda(simple_model, sample_dataset):
    """Test inspecting a model on CUDA if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Move model and data to CUDA
    model = simple_model.cuda()
    dataset = sample_dataset.cuda()

    # When inspecting on CUDA
    modules_info = inspect(model, dataset)

    # Then verify the results
    assert isinstance(modules_info, InspectedModulesInfo)

    modules = modules_info.get_modules()

    assert len(modules) > 0
    assert any(info.forward_called for info in modules)
    assert any(info.total_execution_time > 0 for info in modules)


def test_inspect_with_mixed_precision(sample_dataset):
    """Test inspecting a model with mixed precision."""

    class MixedPrecisionModel(nn.Module):
        def __init__(self):
            super().__init__()
            # First layer in float16
            self.linear1 = nn.Linear(10, 20).to(torch.float16)
            self.relu = nn.ReLU()
            # Second layer in float32
            self.linear2 = nn.Linear(20, 5).to(torch.float32)

        def forward(self, x):
            # Convert input to float16 for first layer
            x = x.to(torch.float16)
            x = self.linear1(x)
            x = self.relu(x)
            # Convert to float32 for second layer
            x = x.to(torch.float32)
            x = self.linear2(x)
            return x

    model = MixedPrecisionModel()

    # When inspecting the model
    modules_info = inspect(model, sample_dataset)

    # Then verify the results
    assert isinstance(modules_info, InspectedModulesInfo)

    modules = modules_info.get_modules()

    assert len(modules) > 0

    # Check precision tracking
    module_info = next(iter(modules))
    precisions = module_info.precisions
    assert torch.float16 in precisions
    assert torch.float32 in precisions


def test_inspect_with_invalid_input():
    """Test inspecting with invalid input."""
    # When inspecting with invalid input
    with pytest.raises(TypeError):
        inspect(None, torch.randn(10, 10))  # pytype: disable=wrong-arg-types


def test_inspect_with_multiple_forward_calls(simple_model, sample_dataset):
    """Test inspecting with multiple forward calls."""
    # When inspecting with multiple forward calls
    modules_info = inspect(simple_model, sample_dataset)

    # Then verify execution tracking
    module_info = next(iter(modules_info.get_modules()))
    assert module_info.execution_count > 0
    assert module_info.total_execution_time > 0
    assert module_info.average_execution_time > 0


def test_inspect_with_custom_object(sample_dataset):
    """Test inspecting a custom object with PyTorch modules."""

    class CustomObject:
        def __init__(self):
            self.model = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
            self.extra_data = torch.randn(5, 5)

        def __call__(self, x):
            return self.model(x)

    custom_obj = CustomObject()

    # When inspecting custom object
    modules_info = inspect(custom_obj, sample_dataset)

    # Then verify the results
    assert isinstance(modules_info, InspectedModulesInfo)

    modules = modules_info.get_modules()

    assert len(modules) > 0
    assert any(info.forward_called for info in modules)


def test_inspect_with_custom_object_without_modules(sample_dataset):
    """Test inspecting a custom object without PyTorch modules."""

    class CustomObject:
        def __init__(self):
            self.data = torch.randn(10, 10)

        def __call__(self, x):
            return x + self.data

    custom_obj = CustomObject()

    # When inspecting custom object without modules
    modules_info = inspect(custom_obj, sample_dataset)

    # Then verify the results
    assert isinstance(modules_info, InspectedModulesInfo)
    assert len(modules_info.get_modules()) == 0  # No modules to inspect


def test_inspect_with_custom_object_with_attributes(sample_dataset):
    """Test inspecting a custom object with various attribute types."""

    class CustomObject:
        def __init__(self):
            self.model = nn.Linear(10, 20)
            self.data = torch.randn(5, 5)
            self.list_data = [1, 2, 3]
            self.dict_data = {"key": "value"}

        def __call__(self, x):
            return self.model(x)

    custom_obj = CustomObject()

    # When inspecting custom object with various attributes
    modules_info = inspect(custom_obj, sample_dataset)

    # Then verify the results
    assert isinstance(modules_info, InspectedModulesInfo)

    modules = modules_info.get_modules()

    assert len(modules) > 0
    assert any(info.forward_called for info in modules)


def test_get_modules_after_inspection(custom_object, sample_dataset):
    """Test getting executed modules after inspection."""
    # When inspecting custom object with various attributes
    modules_info = inspect(custom_object, sample_dataset, number_of_iterations=TEST_NUMBER_OF_ITERATIONS)

    # Then verify the results
    executed_modules = modules_info.get_modules()

    assert len(executed_modules) == 3
    assert executed_modules[0].name == "linear1"
    assert executed_modules[0].execution_count == TEST_NUMBER_OF_ITERATIONS * custom_object.num_iterations
    for idx in [1, 2]:
        assert executed_modules[idx].name in ["linear2", "relu"]
        assert executed_modules[idx].execution_count == TEST_NUMBER_OF_ITERATIONS


def test_get_modules_preserves_discovery_order_by_default():
    """Test getting modules without filtering preserves discovery order."""
    modules_info = InspectedModulesInfo(total_execution_time=1.0, number_of_batches=1)
    for name, execution_time in [("first", 0.1), ("second", 0.3), ("third", 0.2)]:
        modules_info.add_module(
            ModuleInfo(
                module=nn.ReLU(),
                name=name,
                execution_count=1,
                total_execution_time=execution_time,
                object_path=name,
            )
        )

    executed_modules = modules_info.get_modules()

    assert [module.name for module in executed_modules] == ["first", "second", "third"]


def test_get_modules_with_limit_orders_by_execution_time():
    """Test limiting modules selects the modules with the highest execution time."""
    modules_info = InspectedModulesInfo(total_execution_time=1.0, number_of_batches=1)
    for name, execution_time in [("first", 0.1), ("second", 0.3), ("third", 0.2)]:
        modules_info.add_module(
            ModuleInfo(
                module=nn.ReLU(),
                name=name,
                execution_count=1,
                total_execution_time=execution_time,
                object_path=name,
            )
        )

    executed_modules = modules_info.get_modules(limit=2)

    assert [module.name for module in executed_modules] == ["second", "third"]


def test_get_modules_with_min_execution_ratio():
    """Test getting executed modules with a minimum execution ratio."""
    modules_info = InspectedModulesInfo(total_execution_time=1.0, number_of_batches=1)
    modules_info.add_module(
        ModuleInfo(
            module=nn.Linear(10, 10),
            name="linear1",
            execution_count=TEST_NUMBER_OF_ITERATIONS * 4,
            total_execution_time=0.3,
            object_path="linear1",
        )
    )
    modules_info.add_module(
        ModuleInfo(
            module=nn.ReLU(),
            name="relu",
            execution_count=TEST_NUMBER_OF_ITERATIONS,
            total_execution_time=0.2,
            object_path="relu",
        )
    )
    modules_info.add_module(
        ModuleInfo(
            module=nn.Linear(10, 10),
            name="linear2",
            execution_count=TEST_NUMBER_OF_ITERATIONS,
            total_execution_time=0.1,
            object_path="linear2",
        )
    )

    executed_modules = modules_info.get_modules(min_execution_ratio=0.25)

    assert len(executed_modules) == 1
    assert executed_modules[0].name == "linear1"
    assert executed_modules[0].execution_count == TEST_NUMBER_OF_ITERATIONS * 4


@pytest.mark.parametrize("min_execution_ratio", [-0.01, 1.01])
def test_get_modules_rejects_invalid_min_execution_ratio(min_execution_ratio: float):
    """Test rejecting invalid minimum execution ratios."""
    modules_info = InspectedModulesInfo(total_execution_time=1.0, number_of_batches=1)

    with pytest.raises(ValueError, match="value must be between 0 and 1"):
        modules_info.get_modules(min_execution_ratio=min_execution_ratio)


def test_get_modules_after_inspection_with_limit(custom_object, sample_dataset):
    """Test getting executed modules with a limit."""
    # When inspecting custom object with various attributes
    modules_info = inspect(custom_object, sample_dataset, number_of_iterations=TEST_NUMBER_OF_ITERATIONS)

    # Then verify the results
    executed_modules = modules_info.get_modules(limit=2)
    assert len(executed_modules) == 2
    assert executed_modules[0].name == "linear1"
    assert executed_modules[0].execution_count == TEST_NUMBER_OF_ITERATIONS * custom_object.num_iterations

    assert executed_modules[1].name in ["linear2", "relu"]
    assert executed_modules[1].execution_count == TEST_NUMBER_OF_ITERATIONS
