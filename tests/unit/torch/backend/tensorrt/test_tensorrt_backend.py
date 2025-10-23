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
"""Unit tests for TensorRTBackend."""

from typing import cast

import pytest
import torch

from aitune.torch.backend.tensorrt.tensorrt_backend import TensorRTBackend, TensorRTBackendConfig
from aitune.torch.checkpoint.storage_tasks import torch_load_with_custom_types
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_metadata import SampleMetadata
from tests.toy_models.torch_models import ToyTorchModel
from tests.utilities.helpers import requires_cuda

# Constants for testing
IN_FEATURES = 32
OUT_FEATURES = 5
BATCH_SIZE = 2


@pytest.fixture
def mock_tensorrt_components(mocker, tmp_path):
    """Fixture that mocks TensorRT components for testing without actual TensorRT."""
    # Setup mocks
    mock_exporter = mocker.patch("aitune.torch.backend.tensorrt.tensorrt_backend.ONNXExporter")
    mock_builder = mocker.patch("aitune.torch.backend.tensorrt.tensorrt_backend.TensorRTBuilder")
    mock_runtime = mocker.patch("aitune.torch.backend.tensorrt.tensorrt_backend.TensorRTRuntime")

    mock_exporter_instance = mocker.MagicMock()
    mock_exporter_instance.export.return_value = tmp_path / "mock_model.onnx"
    mock_exporter.return_value = mock_exporter_instance

    mock_builder_instance = mocker.MagicMock()
    mock_builder_instance.build.return_value = tmp_path / "mock_model.onnx"

    mock_runtime_instance = mocker.MagicMock()
    mock_runtime_instance.load_engine.return_value = b"mock_engine_bytes"

    # Mock execution context and bindings
    mock_context = mocker.MagicMock()
    mock_bindings = {}
    mock_input_names = ["args_0"]
    mock_output_names = ["outputs_0"]

    # Mock engine info
    mock_engine_info = mocker.MagicMock()
    mock_engine_info.input_names = ["args_0"]
    mock_engine_info.output_names = ["outputs_0"]
    mock_engine_info.input_shapes = {"args_0": (BATCH_SIZE, IN_FEATURES)}
    mock_engine_info.output_shapes = {
        "outputs_0": (BATCH_SIZE, OUT_FEATURES),
    }
    mock_engine_info.input_dtypes = {"args_0": torch.float32}
    mock_engine_info.output_dtypes = {"outputs_0": torch.float32}

    mock_runtime_instance.create_execution_context.return_value = (
        mock_context,
        mock_bindings,
        mock_input_names,
        mock_output_names,
        mock_engine_info,
    )

    mock_runtime.return_value = mock_runtime_instance
    mock_builder.return_value = mock_builder_instance

    return mock_exporter, mock_builder, mock_runtime


def test_tensorrt_backend_config_key():
    """Test backend config with cache_dir."""
    config = TensorRTBackendConfig()
    key1 = config.key()
    key2 = config.key()

    assert key1 == key2


def test_tensorrt_backend_config_describe():
    """Test backend config describe."""
    config = TensorRTBackendConfig()
    describe = config.describe()

    assert describe == "quantization_config=None"

    config = TensorRTBackendConfig(use_dynamo=False)
    describe = config.describe()

    assert describe == "use_dynamo=False,quantization_config=None"


def test_tensorrt_backend_init():
    """Test TensorRTBackend initialization."""
    # Test with default parameters
    backend = TensorRTBackend()

    # Test with custom parameters
    config = TensorRTBackendConfig(
        use_dynamo=True,
        opset_version=16,
    )
    backend = TensorRTBackend(
        config=config,
    )

    assert backend._config is not None
    assert backend._system_monitor is not None
    assert backend._context is None
    assert backend._io_tensors is None
    assert backend._output_names is None
    assert backend._input_names is None
    assert backend._engine_path is None
    assert backend._engine_info is None
    assert backend._cuda_stream is None
    assert backend._start_time is None
    assert backend._end_time is None
    assert backend._outputs is None


@requires_cuda
def test_tensorrt_backend_build(mock_tensorrt_components, tmp_path):
    """Test the build method."""
    mock_exporter, mock_builder, mock_runtime = mock_tensorrt_components

    mock_exporter_instance = mock_exporter.return_value
    mock_builder_instance = mock_builder.return_value
    mock_runtime_instance = mock_runtime.return_value

    # Create backend
    backend = TensorRTBackend()

    # Create model and test data
    device = torch.device("cuda")
    model = ToyTorchModel().to(device).eval()
    samples = model.samples(device=device)
    graph_spec = model.graph_spec(device=device)

    # Build model
    backend = backend.build(model, graph_spec, samples, device=device, cache_dir=tmp_path)
    backend = cast(TensorRTBackend, backend)

    # Verify interactions
    mock_exporter_instance.export.assert_called_once()
    mock_builder_instance.build.assert_called_once()
    mock_runtime_instance.load_engine.assert_called_once()

    assert backend._onnx_path is not None
    assert backend._engine_path is not None

    # Verify backend is properly initialized
    assert backend is not None


@requires_cuda
def test_tensorrt_backend_infer(tmp_path):
    """Test the inference functionality of TensorRTBackend with actual TensorRT.

    This test builds a model with TensorRT and performs actual inference
    to verify the correct functionality of the TensorRT backend.
    """
    # Create backend with default FP16 precision for faster build
    backend = TensorRTBackend()

    # Create model and test data
    device = torch.device("cuda")
    model = ToyTorchModel().to(device).eval()
    samples = model.samples(device=device)
    inputs = model.inputs(device=device)
    graph_spec = model.graph_spec(device=device)

    # Get reference output from the PyTorch model for comparison
    with torch.no_grad():
        reference_outputs = [model(tensor) for tensor in inputs]

    # Build model with TensorRT
    backend = backend.build(model, graph_spec, samples, device=device, cache_dir=tmp_path)
    backend = cast(TensorRTBackend, backend)
    # Note: build() already calls activate() internally

    # Verify backend is properly initialized with actual TensorRT components
    assert backend is not None
    assert backend._context is not None
    assert backend._input_names is not None
    assert backend._output_names is not None

    # Perform inference with TensorRT
    output = backend.infer(inputs[0])

    # Verify output shape and values match the reference PyTorch output
    assert output.shape == (inputs[0].shape[0], OUT_FEATURES)
    # Use a relatively relaxed tolerance due to precision differences
    assert torch.allclose(output, reference_outputs[0], rtol=1e-2, atol=1e-2)


def test_tensorrt_backend_prepare_inputs(mocker):
    """Test the _prepare_inputs method using mocks instead of building an actual model."""
    from aitune.torch.backend.tensorrt.tensorrt_backend import TensorRTBackend

    backend = TensorRTBackend()
    backend._engine_info = mocker.MagicMock()
    backend._context = mocker.MagicMock()
    backend._context.get_tensor_shape.return_value = (1, IN_FEATURES)

    test_tensor = torch.randn(1, IN_FEATURES)

    # case 1: single arg only
    args, kwargs = (test_tensor,), {}
    sample = (args, kwargs)
    backend._engine_info.input_names = ["args_0"]
    input_metadata = SampleMetadata.from_inputs(args, kwargs, strict=True)
    output_metadata = SampleMetadata.from_outputs(sample)
    backend._graph_spec = GraphSpec(name="test_graph", input_spec=input_metadata, output_spec=output_metadata)

    prepared_inputs = backend._prepare_inputs((test_tensor,), {})
    assert torch.equal(prepared_inputs["args_0"], test_tensor)

    # case 2: single kwarg only
    args, kwargs = (), {"x": test_tensor}
    sample = (args, kwargs)
    backend._engine_info.input_names = ["kwargs_x"]
    input_metadata = SampleMetadata.from_inputs(args, kwargs, strict=True)
    output_metadata = SampleMetadata.from_outputs(sample)
    backend._graph_spec = GraphSpec(name="test_graph", input_spec=input_metadata, output_spec=output_metadata)

    prepared_inputs = backend._prepare_inputs((), {"x": test_tensor})
    assert torch.equal(prepared_inputs["kwargs_x"], test_tensor)

    # case 3: single arg and single kwarg
    args, kwargs = (test_tensor,), {"x": test_tensor}
    sample = (args, kwargs)
    backend._engine_info.input_names = ["args_0", "kwargs_x"]
    input_metadata = SampleMetadata.from_inputs(args, kwargs, strict=True)
    output_metadata = SampleMetadata.from_outputs(sample)
    backend._graph_spec = GraphSpec(name="test_graph", input_spec=input_metadata, output_spec=output_metadata)

    prepared_inputs = backend._prepare_inputs((test_tensor,), {"x": test_tensor})
    assert torch.equal(prepared_inputs["args_0"], test_tensor)
    assert torch.equal(prepared_inputs["kwargs_x"], test_tensor)


@requires_cuda
def test_backend_inference_returns_copy_of_tensors(tmp_path):
    """The test validates whether backend returns a copy of tensor and not the view of the allocated output.

    This is important if user invokes several infer functions and the latter don't overwrite the previous results.
    """

    class AddOneModule(torch.nn.Module):
        def forward(self, x):
            return x + 1

    model = AddOneModule()
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    x = torch.tensor([[1, 1]], device=device)
    args, kwargs = (x,), {}
    graph_spec = GraphSpec(
        name="test_graph",
        input_spec=SampleMetadata.from_inputs(args, kwargs),
        output_spec=SampleMetadata.from_outputs(x),
    )

    backend = TensorRTBackend()
    backend = backend.build(model, graph_spec, [(args, kwargs)], device=device, cache_dir=tmp_path)

    output1 = backend.infer(x)
    output2 = backend.infer(x + 1)  # this call should not alter the previous result
    backend.deactivate()

    assert output1.equal(torch.tensor([[2, 2]], device=device))
    assert output2.equal(torch.tensor([[3, 3]], device=device))


@requires_cuda
def test_tensorrt_backend_deactivate(tmp_path):
    """Test the deactivate method."""
    # Create backend
    backend = TensorRTBackend()

    # Create model and test data
    device = torch.device("cuda")
    model = ToyTorchModel().to(device).eval()
    samples = model.samples(device=device)
    graph_spec = model.graph_spec(device=device)

    # Build model
    backend = backend.build(model, graph_spec, samples, device=device, cache_dir=tmp_path)
    backend = cast(TensorRTBackend, backend)
    # Note: build() already calls activate() internally

    # Verify components are set up
    assert backend._context is not None
    assert backend._io_tensors is not None
    assert backend._input_names is not None
    assert backend._output_names is not None
    assert backend._engine_info is not None

    # Test deactivation
    backend.deactivate()
    assert not hasattr(backend, "_context")
    assert not hasattr(backend, "_io_tensors")
    assert not hasattr(backend, "_input_names")
    assert not hasattr(backend, "_output_names")
    assert not hasattr(backend, "_engine_info")
    assert not hasattr(backend, "_cuda_stream")
    assert not hasattr(backend, "_start_time")
    assert not hasattr(backend, "_end_time")
    assert not hasattr(backend, "_outputs")


@requires_cuda
def test_tensorrt_backend_integration(tmp_path):
    """Integration test for TensorRTBackend with actual TensorRT.

    This test is skipped by default and should be run only in environments
    with TensorRT properly installed.
    """
    backend = TensorRTBackend()

    # Create model and test data
    device = torch.device("cuda")
    model = ToyTorchModel().to(device).eval()
    samples = model.samples([BATCH_SIZE], device=device)
    graph_spec = model.graph_spec(device=device)

    # Build model
    backend = backend.build(model, graph_spec, samples, device=device, cache_dir=tmp_path)

    # Verify backend is properly initialized
    assert backend is not None
    # Note: build() already calls activate() internally

    # Test inference
    inputs = model.inputs(device=device)
    output = backend.infer(inputs[0])
    assert output.shape == (BATCH_SIZE, OUT_FEATURES)

    # Test deactivation
    backend.deactivate()


@requires_cuda
def test_build_with_dynamic_shapes(tmp_path, mocker):
    """Test building with dynamic shapes."""
    mock_builder_class = mocker.patch("aitune.torch.backend.tensorrt.tensorrt_backend.TensorRTBuilder")
    mock_builder = mocker.MagicMock()
    mock_runtime_class = mocker.patch("aitune.torch.backend.tensorrt.tensorrt_backend.TensorRTRuntime")
    mock_runtime = mock_runtime_class.return_value

    # Mock the create_execution_context method to return the expected 4 values
    mock_context = mocker.MagicMock()
    mock_bindings = {}
    mock_input_names = ["args_0"]
    mock_output_names = ["outputs_0"]

    # Mock engine info
    mock_engine_info = mocker.MagicMock()
    mock_engine_info.input_names = ["args_0"]
    mock_engine_info.output_names = [
        "outputs_0",
    ]
    mock_engine_info.input_shapes = {"args_0": (BATCH_SIZE, IN_FEATURES)}
    mock_engine_info.output_shapes = {
        "outputs_0": (BATCH_SIZE, OUT_FEATURES),
    }
    mock_engine_info.input_dtypes = {"args_0": torch.float32}
    mock_engine_info.output_dtypes = {"outputs_0": torch.float32}

    mock_runtime.create_execution_context.return_value = (
        mock_context,
        mock_bindings,
        mock_input_names,
        mock_output_names,
        mock_engine_info,
    )

    mock_builder_class.return_value = mock_builder
    mock_runtime_class.return_value = mock_runtime

    # Set dynamic shapes - different values for min, opt, and max
    min_shapes = {"args_0": (1, IN_FEATURES)}
    max_shapes = {"args_0": (4, IN_FEATURES)}

    # Create backend with dynamic shapes
    backend = TensorRTBackend()

    # Create model and test data
    batch_sizes = [1, 2, 4]
    device = torch.device("cuda")

    model = ToyTorchModel().to(device).eval()
    samples = model.samples(batch_sizes=batch_sizes, device=device)
    graph_spec = model.graph_spec(batch_sizes=batch_sizes, device=device)

    # Build the model
    _ = backend.build(model, graph_spec, samples, device=device, cache_dir=tmp_path)

    # Verify builder was initialized with correct dynamic shapes
    mock_builder_class.assert_called_once()
    call_kwargs = mock_builder_class.call_args.kwargs
    assert call_kwargs["min_shapes"] == min_shapes
    assert call_kwargs["opt_shapes"] == max_shapes
    assert call_kwargs["max_shapes"] == max_shapes

    # Verify the TensorRTBuilder.build method was called
    mock_builder.build.assert_called_once()

    # Verify runtime was created
    mock_runtime_class.assert_called_once()

    # Verify runtime was initialized
    mock_runtime.load_engine.assert_called_once()
    mock_runtime.create_execution_context.assert_called_once()


@requires_cuda
def test_build_without_dynamic_shapes(tmp_path, mocker):
    """Test building without dynamic shapes (standard case)."""
    mock_builder_class = mocker.patch("aitune.torch.backend.tensorrt.tensorrt_backend.TensorRTBuilder")
    mock_builder = mock_builder_class.return_value

    mock_runtime_class = mocker.patch("aitune.torch.backend.tensorrt.tensorrt_backend.TensorRTRuntime")
    mock_runtime = mock_runtime_class.return_value

    # Mock the create_execution_context method to return the expected 4 values
    mock_context = mocker.MagicMock()
    mock_bindings = {}
    mock_input_names = ["args_0"]
    mock_output_names = ["outputs_0"]

    # Mock engine info
    mock_engine_info = mocker.MagicMock()
    mock_engine_info.input_names = ["args_0"]
    mock_engine_info.output_names = [
        "outputs_0",
    ]
    mock_engine_info.input_shapes = {"args_0": (BATCH_SIZE, IN_FEATURES)}
    mock_engine_info.output_shapes = {
        "outputs_0": (BATCH_SIZE, OUT_FEATURES),
    }
    mock_engine_info.input_dtypes = {"args_0": torch.float32}
    mock_engine_info.output_dtypes = {"outputs_0": torch.float32}

    mock_runtime.create_execution_context.return_value = (
        mock_context,
        mock_bindings,
        mock_input_names,
        mock_output_names,
        mock_engine_info,
    )

    mock_builder_class.return_value = mock_builder
    mock_runtime_class.return_value = mock_runtime

    # Create backend without setting any dynamic shapes
    backend = TensorRTBackend()

    # Create model and test data
    device = torch.device("cuda")
    model = ToyTorchModel().to(device).eval()
    batch_sizes = [BATCH_SIZE]
    samples = model.samples(batch_sizes=batch_sizes, device=device)
    graph_spec = model.graph_spec(batch_sizes=batch_sizes, device=device)

    # Build the model
    _ = backend.build(model, graph_spec, samples, device=device, cache_dir=tmp_path)

    # Verify builder was initialized with default shape settings (None)
    mock_builder_class.assert_called_once()
    call_kwargs = mock_builder_class.call_args.kwargs
    assert call_kwargs["min_shapes"] == {"args_0": (2, IN_FEATURES)}
    assert call_kwargs["opt_shapes"] == {"args_0": (2, IN_FEATURES)}
    assert call_kwargs["max_shapes"] == {"args_0": (2, IN_FEATURES)}

    # Verify the TensorRTBuilder.build method was called
    mock_builder.build.assert_called_once()

    # Verify runtime was created
    mock_runtime_class.assert_called_once()

    # Verify runtime was initialized
    mock_runtime.load_engine.assert_called_once()
    mock_runtime.create_execution_context.assert_called_once()


@requires_cuda
def test_serialization(tmp_path):
    """Test serialization and deserialization of TensorRTBackend.

    This test verifies that a TensorRTBackend can be serialized to a dictionary
    and deserialized back while maintaining the same inference behavior.
    """
    # Create backend with FP16 precision for faster build
    backend = TensorRTBackend()

    # Create model and test data
    device = torch.device("cuda")
    model = ToyTorchModel().to(device).eval()
    samples = model.samples(device=device)
    inputs = model.inputs(device=device)
    graph_spec = model.graph_spec(device=device)

    # Build model
    backend = backend.build(model, graph_spec, samples, device=device, cache_dir=tmp_path)

    # Get reference output
    reference_output = backend.infer(inputs[0])

    # Serialize to dictionary
    state_dict = backend.to_dict()  # type: ignore

    # Save and load state dict
    torch.save(state_dict, tmp_path / "state_dict.pth")

    # Create new backend from loaded state dict
    model = ToyTorchModel().to("cuda").eval()
    loaded_backend = TensorRTBackend.from_dict(model, torch_load_with_custom_types(tmp_path / "state_dict.pth"))
    loaded_backend.activate()

    # Verify inference results match
    loaded_output = loaded_backend.infer(inputs[0])
    torch.testing.assert_close(reference_output, loaded_output, rtol=1e-2, atol=1e-2)
