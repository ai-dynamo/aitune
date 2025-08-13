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
"""Unit tests for TensorRTBackend custom output object handling."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from aitune.torch.backend.backend import BackendState
from aitune.torch.backend.tensorrt.tensorrt_backend import OutputFormat, TensorRTBackend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_metadata import SampleMetadata

# Test constants
BATCH_SIZE = 2
IN_FEATURES = 32
OUT_FEATURES = 5


class CustomOutput:
    """Custom output class for testing model output handling."""

    def __init__(self, output1=None, output2=None):
        self.output1 = output1
        self.output2 = output2

    def __eq__(self, other):
        if not isinstance(other, CustomOutput):
            return False
        return torch.allclose(self.output1, other.output1) and torch.allclose(self.output2, other.output2)


class CustomOutputModel(nn.Module):
    """Model that returns a custom object with tensor attributes."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(IN_FEATURES, OUT_FEATURES)
        self.linear2 = nn.Linear(IN_FEATURES, OUT_FEATURES * 2)

    def forward(self, x):
        return CustomOutput(self.linear1(x), self.linear2(x))


@pytest.fixture
def mock_graph_spec(mocker):
    """Fixture that mocks GraphSpec for testing."""
    mock_input_metadata = mocker.MagicMock(SampleMetadata)
    # Create a mock _metadata attribute with structure matching SampleMetadata
    mock_input_metadata._metadata = (None, {"input__0": "mock_name"})

    mock_output_metadata = mocker.MagicMock(SampleMetadata)
    # Create a mock _metadata attribute with structure matching SampleMetadata
    mock_output_metadata._metadata = (None, {"output__0": "output1", "output__1": "output2"})

    # Mock tensor_specs to support accessing tensor_specs[0].name and tensor_specs[N].max_shape
    # Create tensor specs for both outputs with proper max_shape attributes
    mock_tensor_spec_0 = mocker.MagicMock()
    mock_tensor_spec_0.name = "output__0"
    mock_tensor_spec_0.max_shape = (BATCH_SIZE, OUT_FEATURES)  # Shape for output__0

    mock_tensor_spec_1 = mocker.MagicMock()
    mock_tensor_spec_1.name = "output__1"
    mock_tensor_spec_1.max_shape = (BATCH_SIZE, OUT_FEATURES * 2)  # Shape for output__1

    mock_output_metadata.tensor_specs = [mock_tensor_spec_0, mock_tensor_spec_1]

    # Mock unflatten_sample method to return the expected structure for CustomOutput
    mock_output_metadata.unflatten_sample.return_value = {
        "output1": mocker.MagicMock(),  # Will be replaced with actual tensor
        "output2": mocker.MagicMock(),  # Will be replaced with actual tensor
    }

    mock_graph_spec = mocker.MagicMock(GraphSpec)
    mock_graph_spec.name = "test_graph"
    mock_graph_spec.input_spec = mock_input_metadata
    mock_graph_spec.output_spec = mock_output_metadata
    return mock_graph_spec


@pytest.fixture
def mock_tensorrt_components(mocker):
    """Fixture that mocks TensorRT components."""
    # Setup mocks
    mock_exporter = mocker.patch("aitune.torch.backend.tensorrt.tensorrt_backend.ONNXExporter")
    mock_model_info = mocker.patch("aitune.torch.backend.tensorrt.torch_model_info.TorchModelInfo")

    # Configure mock model info
    mock_model_info_instance = mocker.MagicMock()
    mock_model_info_instance.output_format = OutputFormat.OBJECT
    mock_model_info_instance.output_class = CustomOutput
    mock_model_info.return_value = mock_model_info_instance

    # Configure mock exporter
    mock_exporter_instance = mocker.MagicMock()
    mock_exporter_instance.export.return_value = "mock_model.onnx"
    mock_exporter.return_value = mock_exporter_instance

    # Configure mock builder
    mock_builder_instance = mocker.MagicMock()
    mock_builder_instance.build.return_value = "mock_model.plan"

    # Configure mock builder
    mock_runtime_instance = mocker.MagicMock()
    mock_runtime_instance.load_engine.return_value = b"mock_engine_bytes"

    # Mock engine info
    mock_engine_info = mocker.MagicMock()
    mock_engine_info.input_names = ["input__0"]
    mock_engine_info.output_names = ["output__0", "output__1"]
    mock_engine_info.input_shapes = {"input__0": (BATCH_SIZE, IN_FEATURES)}
    mock_engine_info.output_shapes = {
        "output__0": (BATCH_SIZE, OUT_FEATURES),
        "output__1": (BATCH_SIZE, OUT_FEATURES * 2),
    }
    mock_engine_info.input_dtypes = {"input__0": torch.float32}
    mock_engine_info.output_dtypes = {"output__0": torch.float32, "output__1": torch.float32}

    # Return mocked components
    return (
        mock_exporter_instance,
        mock_builder_instance,
        mock_runtime_instance,
        mock_engine_info,
        mock_model_info_instance,
    )


@patch("aitune.torch.backend.tensorrt.torch_model_info.TorchModelInfo")
def test_custom_object_output_format(mock_torch_model_info, mock_tensorrt_components, mock_graph_spec, mocker):
    """Test handling of custom object output format."""
    # Setup mock TorchModelInfo
    mock_model_info = MagicMock()
    mock_torch_model_info.return_value = mock_model_info

    # Get mock components
    _, _, _, mock_runtime, mock_engine_info = mock_tensorrt_components

    # Setup mock context and other necessary components
    mock_context = mocker.MagicMock()
    mock_io_tensors = {}
    mock_input_names = ["input__0"]
    mock_output_names = ["output__0", "output__1"]  # Match CustomOutput attributes

    # Configure builder to return our mocks
    mock_runtime.create_execution_context.return_value = (
        mock_context,
        mock_io_tensors,
        mock_input_names,
        mock_output_names,
        mock_engine_info,
    )

    # Create backend
    backend = TensorRTBackend()
    backend._context = mock_context
    backend._io_tensors = mock_io_tensors
    backend._input_names = mock_input_names
    backend._output_names = mock_output_names
    backend._engine_info = mock_engine_info
    backend._torch_model_info = mock_model_info
    backend._output_class = CustomOutput
    backend._output_format = OutputFormat.OBJECT
    # Mock TensorRT context methods
    mock_context.get_tensor_shape.return_value = (BATCH_SIZE, OUT_FEATURES)

    # Create mock outputs
    outputs = {
        "output__0": torch.randn(BATCH_SIZE, OUT_FEATURES),
        "output__1": torch.randn(BATCH_SIZE, OUT_FEATURES * 2),
    }

    # Update unflatten sample output
    mock_graph_spec.output_spec.unflatten_sample.return_value = {
        "output1": outputs["output__0"],
        "output2": outputs["output__1"],
    }

    # Test infer method
    # Mock backend methods using mocker - no context manager needed as mocker handles cleanup
    mocker.patch.object(backend, "_prepare_inputs", return_value={"input__0": torch.randn(BATCH_SIZE, IN_FEATURES)})
    mocker.patch.object(backend, "_set_input_tensors")
    mocker.patch("torch.cuda.stream")  # Patch torch.cuda.stream to avoid CUDA errors

    # Mock other methods needed for inference
    backend._cuda_stream = MagicMock()
    backend._start_time = MagicMock()
    backend._end_time = MagicMock()
    backend._start_time.elapsed_time.return_value = 1.0
    backend._end_time.elapsed_time.return_value = 2.0
    backend._graph_spec = mock_graph_spec

    # Mock execution
    mock_context.execute_async_v3.return_value = True
    backend.state = BackendState.ACTIVE

    # Create mock output allocator with test outputs
    mock_output_allocator = MagicMock()
    mock_output_allocator.outputs = outputs
    backend._output_allocator = mock_output_allocator

    # Mock the unflatten_sample to return the correct mapping for CustomOutput
    def mock_unflatten_sample(outputs_dict):
        return {"output1": outputs_dict["output__0"], "output2": outputs_dict["output__1"]}

    mock_graph_spec.output_spec.unflatten_sample.side_effect = mock_unflatten_sample

    # Mock execution
    mock_context.execute_async_v3.return_value = True

    # Call infer
    result = backend.infer(torch.randn(BATCH_SIZE, IN_FEATURES))

    # Verify result is a CustomOutput instance
    assert isinstance(result, CustomOutput), "Should return a CustomOutput instance"
    assert hasattr(result, "output1"), "Should have output1 attribute"
    assert hasattr(result, "output2"), "Should have output2 attribute"
    assert torch.equal(result.output1, outputs["output__0"]), "output1 values should match"
    assert torch.equal(result.output2, outputs["output__1"]), "output2 values should match"
