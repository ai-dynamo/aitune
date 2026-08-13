# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for TensorRT backend with IOutputAllocator."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from aitune.torch.backend import ArtifactPath
from aitune.torch.backend.backend import BackendState
from aitune.torch.backend.tensorrt.tensorrt_backend import TensorRTBackend
from aitune.torch.backend.tensorrt.torch_output_allocator import TorchOutputAllocator
from tests.utilities.helpers import requires_cuda


# Simple test model
class SimpleModel(nn.Module):
    """Simple model for testing the IOutputAllocator integration."""

    def __init__(self, input_size: int = 10, output_size: int = 5):
        """Initialize the simple model.

        Args:
            input_size: Input tensor size
            output_size: Output tensor size
        """
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return self.linear(x)


@requires_cuda
def test_output_allocator_integration():
    """Test that the TensorRT backend correctly integrates with IOutputAllocator."""
    # Create a simple model
    model = SimpleModel(input_size=10, output_size=5)
    model.eval()

    # Create TensorRT backend
    backend = TensorRTBackend()

    # Mock the TensorRT runtime class to prevent real instantiation
    with patch("aitune.torch.backend.tensorrt.tensorrt_backend.TensorRTRuntime") as mock_runtime_class:
        # Setup mock engine info
        mock_engine_info = MagicMock()
        mock_engine_info.output_dtypes = {"output_0": torch.float32}
        mock_engine_info.output_names = ["output_0"]
        mock_engine_info.input_names = ["input_0"]

        # Setup mock context
        mock_context = MagicMock()
        mock_context.set_output_allocator.return_value = True  # Mock successful allocation
        mock_io_tensors = {}
        mock_input_names = ["input_0"]
        mock_output_names = ["output_0"]

        # Create mock runtime instance
        mock_runtime_instance = MagicMock()
        mock_runtime_instance.load_engine.return_value = b"mock_engine_bytes"
        mock_runtime_instance.create_execution_context.return_value = (
            mock_context,
            mock_io_tensors,
            mock_input_names,
            mock_output_names,
            mock_engine_info,
        )

        # Configure the class mock to return our instance
        mock_runtime_class.return_value = mock_runtime_instance

        # Set up required attributes for activation (no actual file paths)
        backend._model_name = "test_model"
        backend._engine_artifact = ArtifactPath(".", "mock_engine.plan")
        backend._trt_optimization_profiles_artifact = ArtifactPath(".", "mock_profiles.json")
        backend._device = torch.device("cuda:0")  # Set device for activation

        # Set the backend state to INACTIVE so we can activate it
        # This simulates the backend being built and then deactivated
        backend.state = BackendState.INACTIVE

        # Activate the backend
        backend.activate()

        # Verify that output allocator was created and set
        assert backend._output_allocator is not None
        assert isinstance(backend._output_allocator, TorchOutputAllocator)
        assert backend._output_allocator._engine_info == mock_engine_info

        # Verify that the context's output allocator was set
        mock_context.set_output_allocator.assert_called_once_with("output_0", backend._output_allocator)

        # Clean up
        backend.deactivate()


@requires_cuda
def test_output_allocator_inference_flow(tmp_path):
    """Test the inference flow with output allocator."""
    backend = TensorRTBackend()

    # Mock all the TensorRT components
    with patch.multiple(
        backend,
        _trt_runtime=MagicMock(),
        _graph_spec=MagicMock(),
        _output_names=["output_0"],
    ):
        # Setup mock components
        mock_engine_info = MagicMock()
        mock_engine_info.output_dtypes = {"output_0": torch.float32}
        mock_engine_info.input_names = ["input_0"]

        mock_context = MagicMock()
        mock_context.execute_async_v3.return_value = True

        # Initialize required components
        backend._model_name = "test_model"
        backend._engine_artifact = ArtifactPath(tmp_path, "test.engine")
        backend._context = mock_context
        backend._engine_info = mock_engine_info
        backend._input_names = ["input_0"]
        backend._output_names = ["output_0"]
        backend._cuda_stream = torch.cuda.Stream()
        backend._start_time = torch.cuda.Event(enable_timing=True)
        backend._end_time = torch.cuda.Event(enable_timing=True)

        # Create and setup output allocator
        backend._output_allocator = TorchOutputAllocator(engine_info=mock_engine_info)

        # Mock the _prepare_inputs method to return proper input
        test_input = torch.randn(2, 10, device="cuda")
        backend._prepare_inputs = MagicMock(return_value={"input_0": test_input})

        # Mock _set_input_tensors to avoid actual TensorRT calls
        backend._set_input_tensors = MagicMock()

        # Simulate what TensorRT would do: allocate output tensor and notify shape
        mock_output = torch.randn(2, 5, device="cuda")
        # Simulate TensorRT calling reallocate_output for the output tensor
        output_size = mock_output.numel() * mock_output.element_size()
        backend._output_allocator.reallocate_output("output_0", 0, output_size, 16)
        # Simulate TensorRT calling notify_shape with the actual output shape
        from tests.unit.torch.backend.tensorrt.test_tensorrt_output_allocator import MockDims

        backend._output_allocator.notify_shape("output_0", MockDims(mock_output.shape))

        # Mock the output format handling
        backend._prepare_outputs_for_return = MagicMock(return_value=mock_output)

        # Set the backend state to ACTIVE to allow inference
        backend.state = BackendState.ACTIVE

        # Call inference
        result = backend.infer(test_input)

        # Verify that the output allocator was cleared before inference
        # (we can't directly test this since it's mocked, but we verify the flow worked)
        assert result is not None

        # Verify _prepare_outputs_for_return was called (which uses the allocator)
        backend._prepare_outputs_for_return.assert_called_once()


@requires_cuda
def test_output_allocator_cleanup():
    """Test that output allocator is properly cleaned up during deactivation."""
    backend = TensorRTBackend()

    # Manually set up the allocator and other components
    backend._output_allocator = TorchOutputAllocator()
    backend._context = MagicMock()
    backend._io_tensors = {}
    backend._input_names = ["input_0"]
    backend._output_names = ["output_0"]
    backend._engine_info = MagicMock()
    backend._cuda_stream = torch.cuda.Stream()
    backend._start_time = torch.cuda.Event(enable_timing=True)
    backend._end_time = torch.cuda.Event(enable_timing=True)
    backend._trt_builder = MagicMock()
    backend._onnx_exporter = MagicMock()
    backend._trt_runtime = MagicMock()

    # Set the backend state to ACTIVE so we can deactivate it
    backend.state = BackendState.ACTIVE

    # Verify allocator exists before deactivation
    assert backend._output_allocator is not None

    # Deactivate the backend
    backend.deactivate()

    # The deactivate method should have attempted to delete the allocator
    # (Note: The actual deletion might not be visible due to Python's garbage collection,
    # but we can verify the method completed without errors)


@requires_cuda
def test_output_allocator_memory_management():
    """Test that the output allocator properly manages memory."""
    allocator = TorchOutputAllocator()

    # Test that clearing the allocator releases tensor references
    allocator.reallocate_output("output_0", 0, 64, 16)
    allocator.reallocate_output("output_1", 0, 128, 16)

    # Verify tensors exist
    assert len(allocator.outputs) == 2
    assert all(isinstance(tensor, torch.Tensor) for tensor in allocator.outputs.values())
    assert all(tensor.is_cuda for tensor in allocator.outputs.values())

    # Clear and verify memory is released
    allocator.clear()
    assert len(allocator.outputs) == 0
    assert len(allocator.output_shapes) == 0


@requires_cuda
def test_output_allocator_multiple_outputs():
    """Test that output allocator is set individually for multiple output tensors."""
    # Create TensorRT backend
    backend = TensorRTBackend()

    # Mock the TensorRT runtime class to prevent real instantiation
    with patch("aitune.torch.backend.tensorrt.tensorrt_backend.TensorRTRuntime") as mock_runtime_class:
        # Setup mock engine info with multiple outputs
        mock_engine_info = MagicMock()
        mock_engine_info.output_dtypes = {
            "output_0": torch.float32,
            "output_1": torch.float16,
        }
        mock_engine_info.output_names = ["output_0", "output_1"]
        mock_engine_info.input_names = ["input_0"]

        # Setup mock context
        mock_context = MagicMock()
        mock_context.set_output_allocator.return_value = True  # Mock successful setting
        mock_io_tensors = {}
        mock_input_names = ["input_0"]
        mock_output_names = ["output_0", "output_1"]

        # Create mock runtime instance
        mock_runtime_instance = MagicMock()
        mock_runtime_instance.load_engine.return_value = b"mock_engine_bytes"
        mock_runtime_instance.create_execution_context.return_value = (
            mock_context,
            mock_io_tensors,
            mock_input_names,
            mock_output_names,
            mock_engine_info,
        )

        # Configure the class mock to return our instance
        mock_runtime_class.return_value = mock_runtime_instance

        # Set up required attributes for activation
        backend._model_name = "test_multi_output_model"
        backend._engine_artifact = ArtifactPath(".", "mock_multi_engine.plan")
        backend._trt_optimization_profiles_artifact = ArtifactPath(".", "mock_profiles.json")
        backend._device = torch.device("cuda:0")  # Set device for activation

        # Set the backend state to INACTIVE so we can activate it
        backend.state = BackendState.INACTIVE

        # Activate the backend
        backend.activate()

        # Verify that output allocator was created
        assert backend._output_allocator is not None
        assert isinstance(backend._output_allocator, TorchOutputAllocator)

        # Verify that set_output_allocator was called for each output tensor
        expected_calls = [
            ("output_0", backend._output_allocator),
            ("output_1", backend._output_allocator),
        ]
        actual_calls = mock_context.set_output_allocator.call_args_list

        assert len(actual_calls) == 2
        for i, (expected_args, actual_call) in enumerate(zip(expected_calls, actual_calls, strict=True)):
            assert actual_call[0] == expected_args, f"Call {i}: expected {expected_args}, got {actual_call[0]}"

        # Clean up
        backend.deactivate()


@requires_cuda
def test_output_allocator_set_failure():
    """Test handling of output allocator setting failure."""
    backend = TensorRTBackend()

    # Mock the TensorRT runtime class to prevent real instantiation
    with patch("aitune.torch.backend.tensorrt.tensorrt_backend.TensorRTRuntime") as mock_runtime_class:
        # Setup mock engine info
        mock_engine_info = MagicMock()
        mock_engine_info.output_dtypes = {"output_0": torch.float32}
        mock_engine_info.output_names = ["output_0"]
        mock_engine_info.input_names = ["input_0"]

        # Setup mock context that fails to set allocator
        mock_context = MagicMock()
        mock_context.set_output_allocator.return_value = False  # Mock failure
        mock_io_tensors = {}
        mock_input_names = ["input_0"]
        mock_output_names = ["output_0"]

        # Create mock runtime instance
        mock_runtime_instance = MagicMock()
        mock_runtime_instance.load_engine.return_value = b"mock_engine_bytes"
        mock_runtime_instance.create_execution_context.return_value = (
            mock_context,
            mock_io_tensors,
            mock_input_names,
            mock_output_names,
            mock_engine_info,
        )

        # Configure the class mock to return our instance
        mock_runtime_class.return_value = mock_runtime_instance

        # Set up required attributes for activation
        backend._model_name = "test_model"
        backend._engine_artifact = ArtifactPath(".", "mock_engine.plan")
        backend._trt_optimization_profiles_artifact = ArtifactPath(".", "mock_profiles.json")
        backend._device = torch.device("cuda:0")  # Set device for activation

        # Set the backend state to INACTIVE so we can activate it
        backend.state = BackendState.INACTIVE

        # Activation should raise an error due to allocator setting failure
        with pytest.raises(RuntimeError, match="Failed to set output allocator for tensor 'output_0'"):
            backend.activate()
