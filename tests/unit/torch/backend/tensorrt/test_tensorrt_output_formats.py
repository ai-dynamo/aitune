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
"""Unit tests for TensorRTBackend output formats and tensor handling."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from aitune.torch.backend.tensorrt.tensorrt_backend import OutputFormat, TensorRTBackend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_metadata import SampleMetadata
from tests.utilities.helpers import requires_cuda

# Test constants
BATCH_SIZE = 2
IN_FEATURES = 32
OUT_FEATURES = 5


class GraphSpecModule(nn.Module):
    """Mixin for models that return a graph spec."""

    def graph_spec(self, samples):
        """Get graph spec for the model."""
        graph_spec = None
        for sample in samples:
            args, kwargs = sample
            outputs = self(*args, **kwargs)
            input_metadata = SampleMetadata.from_sample(sample, prefix="input")
            output_metadata = SampleMetadata.from_sample(outputs, prefix="output")
            if graph_spec is None:
                graph_spec = GraphSpec(name="toy_model", input_spec=input_metadata, output_spec=output_metadata)
            else:
                graph_spec.update_shapes_seen(input_metadata, output_metadata)
        return graph_spec


class TensorOutputModel(GraphSpecModule):
    """Model that returns a single tensor."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(IN_FEATURES, OUT_FEATURES)

    def forward(self, x):
        return self.linear(x)


class TupleOutputModel(GraphSpecModule):
    """Model that returns a tuple of tensors."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(IN_FEATURES, OUT_FEATURES)
        self.linear2 = nn.Linear(IN_FEATURES, OUT_FEATURES * 2)

    def forward(self, x):
        return self.linear1(x), self.linear2(x)


class ListOutputModel(GraphSpecModule):
    """Model that returns a list of tensors."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(IN_FEATURES, OUT_FEATURES)
        self.linear2 = nn.Linear(IN_FEATURES, OUT_FEATURES * 2)

    def forward(self, x):
        return [self.linear1(x), self.linear2(x)]


class DictOutputModel(GraphSpecModule):
    """Model that returns a dictionary of tensors."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(IN_FEATURES, OUT_FEATURES)
        self.linear2 = nn.Linear(IN_FEATURES, OUT_FEATURES * 2)

    def forward(self, x):
        return {"output1": self.linear1(x), "output2": self.linear2(x)}


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

    mock_exporter_instance = mocker.MagicMock()
    mock_exporter_instance.export.return_value = "mock_model.onnx"
    mock_exporter.return_value = mock_exporter_instance

    mock_builder_instance = mocker.MagicMock()
    mock_builder_instance.build.return_value = "mock_model.plan"
    mock_builder_instance.load_engine.return_value = b"mock_engine_bytes"

    # Mock engine info
    mock_engine_info = mocker.MagicMock()
    mock_engine_info.input_names = ["input__0"]
    mock_engine_info.output_names = ["output__0"]
    mock_engine_info.input_shapes = {"input__0": (BATCH_SIZE, IN_FEATURES)}
    mock_engine_info.output_shapes = {
        "output__0": (BATCH_SIZE, OUT_FEATURES),
    }
    mock_engine_info.input_dtypes = {"input__0": torch.float32}
    mock_engine_info.output_dtypes = {"output__0": torch.float32}

    # Return mocked components
    return mock_exporter_instance, mock_builder_instance, mock_engine_info


@requires_cuda
@patch("aitune.torch.backend.tensorrt.torch_model_info.TorchModelInfo")
def test_non_contiguous_input_tensor(mock_torch_model_info, mock_tensorrt_components, mocker):
    """Test handling of non-contiguous input tensors."""
    # Get mock components
    _, mock_builder, mock_engine_info = mock_tensorrt_components

    # Setup mock context
    mock_context = mocker.MagicMock()
    mock_context.get_tensor_shape.return_value = (BATCH_SIZE, OUT_FEATURES)

    # Setup mock tensors
    mock_io_tensors = {}
    mock_input_names = ["input__0"]
    mock_output_names = ["output__0"]

    # Configure builder to return our mocks
    mock_builder.create_execution_context.return_value = (
        mock_context,
        mock_io_tensors,
        mock_input_names,
        mock_output_names,
        mock_engine_info,
    )

    # Create backend and set required attributes
    backend = TensorRTBackend()
    backend._context = mock_context
    backend._io_tensors = mock_io_tensors
    backend._input_names = mock_input_names
    backend._output_names = mock_output_names
    backend._engine_info = mock_engine_info

    # Create a non-contiguous tensor by transposing a tensor twice
    # e.g., transpose(0,1) makes it non-contiguous but preserves shape
    original_tensor = torch.randn(IN_FEATURES, BATCH_SIZE)
    non_contiguous_tensor = original_tensor.transpose(0, 1)
    assert not non_contiguous_tensor.is_contiguous(), "Test tensor should be non-contiguous"

    # Mock _prepare_inputs to return our non-contiguous tensor
    inputs = {"input": non_contiguous_tensor}
    with patch.object(backend, "_prepare_inputs", return_value=inputs):
        # Call _set_input_tensors directly to test contiguity handling
        backend._set_input_tensors(inputs)

        # Verify context.set_tensor_address was called with tensor.data_ptr()
        mock_context.set_tensor_address.assert_called_once()

        # The input tensor should have been made contiguous
        args = mock_context.set_input_shape.call_args
        assert args is not None, "set_input_shape should have been called"
        assert inputs["input"].is_contiguous(), "Input tensor should be contiguous"


@requires_cuda
@patch("aitune.torch.backend.tensorrt.torch_model_info.TorchModelInfo")
def test_cpu_to_cuda_tensor_conversion(mock_torch_model_info, mock_tensorrt_components, mocker):
    """Test automatic conversion of CPU tensors to CUDA."""
    # Get mock components
    _, mock_builder, mock_engine_info = mock_tensorrt_components

    # Setup mock context
    mock_context = mocker.MagicMock()
    mock_context.get_tensor_shape.return_value = (BATCH_SIZE, OUT_FEATURES)

    # Setup mock tensors
    mock_io_tensors = {}
    mock_input_names = ["input__0"]
    mock_output_names = ["output__0"]

    # Configure builder to return our mocks
    mock_builder.create_execution_context.return_value = (
        mock_context,
        mock_io_tensors,
        mock_input_names,
        mock_output_names,
        mock_engine_info,
    )

    # Create backend and set required attributes
    backend = TensorRTBackend()
    backend._context = mock_context
    backend._io_tensors = mock_io_tensors
    backend._input_names = mock_input_names
    backend._output_names = mock_output_names
    backend._engine_info = mock_engine_info

    # Create tensor on CPU
    cpu_tensor = torch.randn(BATCH_SIZE, IN_FEATURES)
    assert not cpu_tensor.is_cuda, "Test tensor should be on CPU"

    # Call _set_input_tensors to test device conversion
    inputs = {"input__0": cpu_tensor}
    with patch.object(backend, "_prepare_inputs", return_value=inputs):
        backend._set_input_tensors(inputs)

        # Verify context.set_tensor_address was called
        mock_context.set_tensor_address.assert_called_once()
        assert inputs["input__0"].is_cuda, "Input tensor should be on CUDA"


# Test different output formats with mocked TorchModelInfo
@requires_cuda
@patch("aitune.torch.backend.tensorrt.torch_model_info.TorchModelInfo")
def test_tensor_output_format(mock_torch_model_info, mock_tensorrt_components, mock_graph_spec, mocker):
    """Test handling of single tensor output format."""
    # Setup mock TorchModelInfo
    mock_model_info = MagicMock()
    mock_torch_model_info.return_value = mock_model_info

    # Get mock components
    _, mock_builder, mock_engine_info = mock_tensorrt_components

    # Setup mock context and other necessary components
    mock_context = mocker.MagicMock()
    mock_io_tensors = {}
    mock_input_names = ["input__0"]
    mock_output_names = ["output__0"]

    # Configure builder to return our mocks
    mock_builder.create_execution_context.return_value = (
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
    backend._graph_spec = mock_graph_spec
    backend._output_format = OutputFormat.TENSOR
    backend._device = torch.device("cuda:0")

    # Mock TensorRT context methods
    mock_context.get_tensor_shape.return_value = (BATCH_SIZE, OUT_FEATURES)

    # Create mock output allocator with test outputs
    mock_output_allocator = MagicMock()
    output_tensor = torch.randn(BATCH_SIZE, OUT_FEATURES)
    mock_output_allocator.outputs = {"output__0": output_tensor}
    backend._output_allocator = mock_output_allocator

    outputs_copy = backend._prepare_outputs_for_return()

    # Verify that a single tensor is returned for TENSOR format
    assert isinstance(outputs_copy, torch.Tensor), "Should return a single tensor"


@requires_cuda
@patch("aitune.torch.backend.tensorrt.torch_model_info.TorchModelInfo")
def test_tuple_output_format(mock_torch_model_info, mock_tensorrt_components, mock_graph_spec, mocker):
    """Test handling of tuple output format."""
    # Setup mock TorchModelInfo
    mock_model_info = MagicMock()
    mock_torch_model_info.return_value = mock_model_info

    # Get mock components
    _, mock_builder, mock_engine_info = mock_tensorrt_components

    # Override engine info
    mock_engine_info.output_names = ["output__0", "output__1"]
    mock_engine_info.output_shapes = {
        "output__0": (BATCH_SIZE, OUT_FEATURES),
        "output__1": (BATCH_SIZE, OUT_FEATURES * 2),
    }
    mock_engine_info.output_dtypes = {"output__0": torch.float32, "output__1": torch.float32}

    # Setup mock context and other necessary components
    mock_context = mocker.MagicMock()
    mock_io_tensors = {}
    mock_input_names = ["input__0"]
    mock_output_names = ["output__0", "output__1"]  # Two outputs

    # Configure builder to return our mocks
    mock_builder.create_execution_context.return_value = (
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
    backend._graph_spec = mock_graph_spec

    backend._output_format = OutputFormat.TUPLE
    # Mock TensorRT context methods
    mock_context.get_tensor_shape.return_value = (BATCH_SIZE, OUT_FEATURES)

    # Create mock output allocator with test outputs
    mock_output_allocator = MagicMock()
    output_tensor_0 = torch.randn(BATCH_SIZE, OUT_FEATURES)
    output_tensor_1 = torch.randn(BATCH_SIZE, OUT_FEATURES * 2)
    mock_output_allocator.outputs = {"output__0": output_tensor_0, "output__1": output_tensor_1}
    backend._output_allocator = mock_output_allocator
    mock_graph_spec.output_spec.unflatten_sample.return_value = (output_tensor_0, output_tensor_1)

    outputs_copy = backend._prepare_outputs_for_return()

    # Verify that a tuple is returned for TUPLE format
    assert isinstance(outputs_copy, tuple), "Should return a tuple"
    assert len(outputs_copy) == 2, "Should have 2 outputs in the tuple"


@requires_cuda
@patch("aitune.torch.backend.tensorrt.torch_model_info.TorchModelInfo")
def test_list_output_format(mock_torch_model_info, mock_tensorrt_components, mock_graph_spec, mocker):
    """Test handling of list output format."""
    # Setup mock TorchModelInfo
    mock_model_info = MagicMock()
    mock_torch_model_info.return_value = mock_model_info

    # Get mock components
    _, mock_builder, mock_engine_info = mock_tensorrt_components

    # Override engine info
    mock_engine_info.output_names = ["output__0", "output__1"]
    mock_engine_info.output_shapes = {
        "output__0": (BATCH_SIZE, OUT_FEATURES),
        "output__1": (BATCH_SIZE, OUT_FEATURES * 2),
    }
    mock_engine_info.output_dtypes = {"output__0": torch.float32, "output__1": torch.float32}

    # Setup mock context and other necessary components
    mock_context = mocker.MagicMock()
    mock_io_tensors = {}
    mock_input_names = ["input__0"]
    mock_output_names = ["output__0", "output__1"]  # Two outputs

    # Configure builder to return our mocks
    mock_builder.create_execution_context.return_value = (
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
    backend._graph_spec = mock_graph_spec
    backend._output_format = OutputFormat.LIST
    backend._device = torch.device("cuda:0")

    # Mock TensorRT context methods
    mock_context.get_tensor_shape.return_value = (BATCH_SIZE, OUT_FEATURES)

    # Create mock output allocator with test outputs
    mock_output_allocator = MagicMock()
    output_tensor_0 = torch.randn(BATCH_SIZE, OUT_FEATURES)
    output_tensor_1 = torch.randn(BATCH_SIZE, OUT_FEATURES * 2)
    mock_output_allocator.outputs = {"output__0": output_tensor_0, "output__1": output_tensor_1}
    backend._output_allocator = mock_output_allocator

    outputs_copy = backend._prepare_outputs_for_return()

    # Verify that a list is returned for LIST format
    assert isinstance(outputs_copy, list), "Should return a list"
    assert len(outputs_copy) == 2, "Should have 2 outputs in the list"


@requires_cuda
@patch("aitune.torch.backend.tensorrt.torch_model_info.TorchModelInfo")
def test_dict_output_format(mock_torch_model_info, mock_tensorrt_components, mock_graph_spec, mocker):
    """Test handling of dictionary output format."""
    # Setup mock TorchModelInfo
    mock_model_info = MagicMock()
    mock_model_info.output_format = OutputFormat.DICT
    mock_torch_model_info.return_value = mock_model_info

    # Get mock components
    _, mock_builder, mock_engine_info = mock_tensorrt_components

    # Override engine info
    mock_engine_info.output_names = ["output__0", "output__1"]
    mock_engine_info.output_shapes = {
        "output__0": (BATCH_SIZE, OUT_FEATURES),
        "output__1": (BATCH_SIZE, OUT_FEATURES * 2),
    }
    mock_engine_info.output_dtypes = {"output__0": torch.float32, "output__1": torch.float32}

    # Setup mock context and other necessary components
    mock_context = mocker.MagicMock()
    mock_io_tensors = {}
    mock_input_names = ["input__0"]
    mock_output_names = ["output__0", "output__1"]  # Two outputs

    # Configure builder to return our mocks
    mock_builder.create_execution_context.return_value = (
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
    backend._graph_spec = mock_graph_spec
    # Set output_format which is normally set during build() method
    backend._output_format = OutputFormat.DICT

    # Mock TensorRT context methods
    mock_context.get_tensor_shape.return_value = (BATCH_SIZE, OUT_FEATURES)

    # Create mock output allocator with test outputs
    mock_output_allocator = MagicMock()
    output_tensor_0 = torch.randn(BATCH_SIZE, OUT_FEATURES)
    output_tensor_1 = torch.randn(BATCH_SIZE, OUT_FEATURES * 2)
    mock_output_allocator.outputs = {"output__0": output_tensor_0, "output__1": output_tensor_1}
    backend._output_allocator = mock_output_allocator

    outputs_copy = backend._prepare_outputs_for_return()

    # Verify that a dict is returned for DICT format
    assert isinstance(outputs_copy, dict), "Should return a dictionary"
    assert len(outputs_copy) == 2, "Should have 2 outputs in the dictionary"


@requires_cuda
def test_integration_tensor_output(tmp_path):
    """Integration test for single tensor output format."""
    # Create model and data
    device = torch.device("cuda")
    model = TensorOutputModel().to(device).eval()
    test_tensor = torch.randn(BATCH_SIZE, IN_FEATURES, device=device)
    samples = [((test_tensor,), {})]
    graph_spec = model.graph_spec(samples=samples)

    # Build with TensorRT
    backend = TensorRTBackend()
    backend = backend.build(module=model, graph_spec=graph_spec, data=samples, device=device, cache_dir=tmp_path)
    # Note: build() already calls activate() internally

    # Verify output is a tensor
    output = backend.infer(test_tensor)
    assert isinstance(output, torch.Tensor), "Output should be a tensor"
    assert output.shape == (BATCH_SIZE, OUT_FEATURES), "Output shape incorrect"

    backend.deactivate()


@requires_cuda
def test_integration_tuple_output(tmp_path):
    """Integration test for tuple output format."""
    # Create model and data
    device = torch.device("cuda")
    model = TupleOutputModel().to(device).eval()
    test_tensor = torch.randn(BATCH_SIZE, IN_FEATURES, device=device)
    samples = [((test_tensor,), {})]
    graph_spec = model.graph_spec(samples=samples)

    # Build with TensorRT
    backend = TensorRTBackend()
    backend = backend.build(module=model, graph_spec=graph_spec, data=samples, device=device, cache_dir=tmp_path)
    # Note: build() already calls activate() internally

    # Verify output is a tuple
    output = backend.infer(test_tensor)
    assert isinstance(output, tuple), "Output should be a tuple"
    assert len(output) == 2, "Output should have 2 tensors"
    assert output[0].shape == (BATCH_SIZE, OUT_FEATURES), "First output shape incorrect"
    assert output[1].shape == (BATCH_SIZE, OUT_FEATURES * 2), "Second output shape incorrect"

    backend.deactivate()


@requires_cuda
def test_integration_list_output(tmp_path):
    """Integration test for list output format."""
    # Create model and data
    device = torch.device("cuda")
    model = ListOutputModel().to(device).eval()
    test_tensor = torch.randn(BATCH_SIZE, IN_FEATURES, device=device)
    samples = [((test_tensor,), {})]
    graph_spec = model.graph_spec(samples=samples)

    # Build with TensorRT
    backend = TensorRTBackend()
    backend = backend.build(module=model, graph_spec=graph_spec, data=samples, device=device, cache_dir=tmp_path)
    # Note: build() already calls activate() internally

    # Verify output is a list
    output = backend.infer(test_tensor)
    assert isinstance(output, list), "Output should be a list"
    assert len(output) == 2, "Output should have 2 tensors"
    assert output[0].shape == (BATCH_SIZE, OUT_FEATURES), "First output shape incorrect"
    assert output[1].shape == (BATCH_SIZE, OUT_FEATURES * 2), "Second output shape incorrect"

    backend.deactivate()


@requires_cuda
def test_integration_dict_output(tmp_path):
    """Integration test for dictionary output format."""
    # Create model and data
    device = torch.device("cuda")
    model = DictOutputModel().to(device).eval()
    test_tensor = torch.randn(BATCH_SIZE, IN_FEATURES, device=device)
    samples = [((test_tensor,), {})]
    graph_spec = model.graph_spec(samples=samples)

    # Build with TensorRT
    backend = TensorRTBackend()
    backend = backend.build(module=model, graph_spec=graph_spec, data=samples, device=device, cache_dir=tmp_path)
    # Note: build() already calls activate() internally

    # Verify output is a dictionary
    output = backend.infer(test_tensor)
    assert isinstance(output, dict), "Output should be a dictionary"
    assert len(output) == 2, "Output should have 2 tensors"
    assert "output1" in output, "Missing output1 key"
    assert "output2" in output, "Missing output2 key"
    assert output["output1"].shape == (BATCH_SIZE, OUT_FEATURES), "First output shape incorrect"
    assert output["output2"].shape == (BATCH_SIZE, OUT_FEATURES * 2), "Second output shape incorrect"

    backend.deactivate()


@requires_cuda
def test_non_contiguous_input_integration(tmp_path):
    """Integration test with non-contiguous input tensor."""
    # Create model and data
    device = torch.device("cuda")
    model = TensorOutputModel().to(device).eval()
    contiguous_tensor = torch.randn(BATCH_SIZE, IN_FEATURES, device=device)
    samples = [((contiguous_tensor,), {})]
    graph_spec = model.graph_spec(samples=samples)

    # Create a non-contiguous tensor with the correct dimensions
    # Create a tensor with swapped dimensions and transpose to get the right shape but non-contiguous
    non_contiguous_tensor = torch.randn(IN_FEATURES, BATCH_SIZE, device="cuda").transpose(0, 1)
    assert not non_contiguous_tensor.is_contiguous(), "Test tensor should be non-contiguous"
    assert non_contiguous_tensor.shape == (BATCH_SIZE, IN_FEATURES), "Non-contiguous tensor should have correct shape"

    # Build with TensorRT
    backend = TensorRTBackend()
    backend = backend.build(
        module=model,
        graph_spec=graph_spec,
        data=samples,
        device=device,
        cache_dir=tmp_path,
    )
    # Note: build() already calls activate() internally

    # Infer with non-contiguous tensor
    output = backend.infer(non_contiguous_tensor)

    # Verify output shape is correct
    assert output.shape == (BATCH_SIZE, OUT_FEATURES), "Output shape incorrect"
    assert output.is_cuda, "Output should be on CUDA"

    backend.deactivate()


@requires_cuda
def test_cpu_input_integration(tmp_path):
    """Integration test with CPU input tensor that gets moved to CUDA."""
    # Create model and data
    device = torch.device("cuda")
    model = TensorOutputModel().to(device).eval()
    cuda_tensor = torch.randn(BATCH_SIZE, IN_FEATURES, device=device)
    samples = [((cuda_tensor,), {})]
    graph_spec = model.graph_spec(samples=samples)

    # Create a CPU tensor with same values
    cpu_tensor = cuda_tensor.cpu()
    assert not cpu_tensor.is_cuda, "Test tensor should be on CPU"

    # Build with TensorRT
    backend = TensorRTBackend()
    backend = backend.build(module=model, graph_spec=graph_spec, data=samples, device=device, cache_dir=tmp_path)
    # Note: build() already calls activate() internally

    # Infer with CPU tensor (should be automatically moved to CUDA)
    output = backend.infer(cpu_tensor)

    # Verify output shape is correct
    assert output.shape == (BATCH_SIZE, OUT_FEATURES), "Output shape incorrect"
    assert output.is_cuda, "Output should be on CUDA"

    backend.deactivate()
