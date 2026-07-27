# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TensorRTBackend."""

from typing import cast

import pytest
import torch
from polygraphy.backend.trt import Profile

from aitune.exceptions import AITuneUserInputError
from aitune.torch.backend.tensorrt.tensorrt_backend import ProfileMode, TensorRTBackend, TensorRTBackendConfig
from aitune.torch.backend.tensorrt.tensorrt_profile import TensorRTProfile
from aitune.torch.checkpoint.storage_tasks import torch_load_with_custom_types
from aitune.torch.dynamic_shapes import BatchDim
from aitune.torch.utils.tensor import format_tensor_name
from tests.toy_models.torch_models import ToyTorchModel
from tests.utilities.helpers import make_graph_spec, requires_cuda, update_input_spec

# Constants for testing
IN_FEATURES = 32
OUT_FEATURES = 5
BATCH_SIZE = 2


def _single_input(x):
    return x


def _single_input_with_options(x, **kwargs):
    return x, kwargs


def _two_inputs(arg, x):
    return arg, x


def _profile_inputs(x, input_tensor):
    return x, input_tensor


@pytest.fixture
def mock_tensorrt_components(mocker, tmp_path):
    """Fixture that mocks TensorRT components for testing without actual TensorRT."""
    # Setup mocks
    mock_exporter = mocker.patch("aitune.torch.backend.tensorrt.tensorrt_backend.ONNXExporter")
    mock_builder = mocker.patch("aitune.torch.backend.tensorrt.tensorrt_backend.TensorRTBuilder")
    mock_runtime = mocker.patch("aitune.torch.backend.tensorrt.tensorrt_backend.TensorRTRuntime")

    # Create dummy build artifacts so .stat().st_size works after mocked builds
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model.onnx").write_bytes(b"fake")
    (tmp_path / "tensorrt").mkdir()
    (tmp_path / "tensorrt" / "model.plan").write_bytes(b"fake")

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
    mock_input_names = ["x"]
    mock_output_names = ["outputs_0"]

    # Mock engine info
    mock_engine_info = mocker.MagicMock()
    mock_engine_info.input_names = ["x"]
    mock_engine_info.output_names = ["outputs_0"]
    mock_engine_info.input_shapes = {"x": (BATCH_SIZE, IN_FEATURES)}
    mock_engine_info.output_shapes = {
        "outputs_0": (BATCH_SIZE, OUT_FEATURES),
    }
    mock_engine_info.input_dtypes = {"x": torch.float32}
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

    assert backend._config is not None
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


def test_tensorrt_backend_init_with_custom_parameters():
    """Test TensorRTBackend initialization with custom parameters."""
    config = TensorRTBackendConfig(
        use_dynamo=True,
        opset_version=16,
    )
    backend = TensorRTBackend(
        config=config,
    )

    assert backend._config is not None
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
    x_name = format_tensor_name("x", "input")
    args, kwargs = (test_tensor,), {}
    sample = (args, kwargs)
    backend._engine_info.input_names = [x_name]
    backend._graph_spec = make_graph_spec(_single_input, sample, sample)

    prepared_inputs = backend._prepare_inputs((test_tensor,), {})
    assert torch.equal(prepared_inputs[x_name], test_tensor)

    # case 2: single kwarg only
    args, kwargs = (), {"x": test_tensor}
    sample = (args, kwargs)
    backend._engine_info.input_names = [x_name]
    backend._graph_spec = make_graph_spec(_single_input, sample, sample)

    prepared_inputs = backend._prepare_inputs((), {"x": test_tensor})
    assert torch.equal(prepared_inputs[x_name], test_tensor)

    # case 3: single arg and single kwarg
    args, kwargs = (test_tensor,), {"x": test_tensor}
    sample = (args, kwargs)
    arg_name = format_tensor_name("arg", "input")
    backend._engine_info.input_names = [arg_name, x_name]
    backend._graph_spec = make_graph_spec(_two_inputs, sample, sample)

    prepared_inputs = backend._prepare_inputs((test_tensor,), {"x": test_tensor})
    assert torch.equal(prepared_inputs[arg_name], test_tensor)
    assert torch.equal(prepared_inputs[x_name], test_tensor)


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
    graph_spec = make_graph_spec(model.forward, (args, kwargs), x)

    backend = TensorRTBackend()
    backend = backend.build(model, graph_spec, [(args, kwargs)], device=device, cache_dir=tmp_path)

    output1 = backend.infer(x)
    output2 = backend.infer(x + 1)  # this call should not alter the previous result
    backend.deactivate()

    assert output1.equal(torch.tensor([[2, 2]], device=device))
    assert output2.equal(torch.tensor([[3, 3]], device=device))


@requires_cuda
@pytest.mark.parametrize("use_dynamo", [True, False], ids=["use_dynamo_true", "use_dynamo_false"])
def test_backend_handle_args_kwargs(tmp_path, use_dynamo):
    """The test validates backend against args and kwargs handling."""

    class TestModule(torch.nn.Module):
        def forward(self, x, y):
            return x + 1, y + 2

    model = TestModule()
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    x = torch.tensor([[1, 1]], device=device)
    y = torch.tensor([[1, 1]], device=device)
    args, kwargs = (x,), {"y": y}
    graph_spec = make_graph_spec(model.forward, (args, kwargs), (x, y))

    config = TensorRTBackendConfig(use_dynamo=use_dynamo)
    backend = TensorRTBackend(config=config)
    backend = backend.build(model, graph_spec, [(args, kwargs)], device=device, cache_dir=tmp_path)

    res_x, res_y = backend.infer(x, y=y)
    backend.deactivate()

    assert res_x.equal(torch.tensor([[2, 2]], device=device))
    assert res_y.equal(torch.tensor([[3, 3]], device=device))


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
    mock_input_names = ["x"]
    mock_output_names = ["outputs_0"]

    # Mock engine info
    mock_engine_info = mocker.MagicMock()
    mock_engine_info.input_names = ["x"]
    mock_engine_info.output_names = [
        "outputs_0",
    ]
    mock_engine_info.input_shapes = {"x": (BATCH_SIZE, IN_FEATURES)}
    mock_engine_info.output_shapes = {
        "outputs_0": (BATCH_SIZE, OUT_FEATURES),
    }
    mock_engine_info.input_dtypes = {"x": torch.float32}
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

    # Create dummy engine file so .stat().st_size works after mocked build
    (tmp_path / "tensorrt").mkdir()
    (tmp_path / "tensorrt" / "model.plan").write_bytes(b"fake")

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

    # Verify builder was initialized with profiles
    mock_builder_class.assert_called_once()
    call_kwargs = mock_builder_class.call_args.kwargs
    assert "profiles" in call_kwargs
    assert isinstance(call_kwargs["profiles"], list)
    assert len(call_kwargs["profiles"]) == 1  # Single profile with min/max range

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
    mock_input_names = ["x"]
    mock_output_names = ["outputs_0"]

    # Mock engine info
    mock_engine_info = mocker.MagicMock()
    mock_engine_info.input_names = ["x"]
    mock_engine_info.output_names = [
        "outputs_0",
    ]
    mock_engine_info.input_shapes = {"x": (BATCH_SIZE, IN_FEATURES)}
    mock_engine_info.output_shapes = {
        "outputs_0": (BATCH_SIZE, OUT_FEATURES),
    }
    mock_engine_info.input_dtypes = {"x": torch.float32}
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

    # Create dummy engine file so .stat().st_size works after mocked build
    (tmp_path / "tensorrt").mkdir()
    (tmp_path / "tensorrt" / "model.plan").write_bytes(b"fake")

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

    # Verify builder was initialized with profiles
    mock_builder_class.assert_called_once()
    call_kwargs = mock_builder_class.call_args.kwargs
    assert "profiles" in call_kwargs
    assert isinstance(call_kwargs["profiles"], list)
    assert len(call_kwargs["profiles"]) == 1  # Single profile for single batch size

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


def test_get_profiles_single_profile():
    """Test the _get_profiles_from_shapes method."""
    backend = TensorRTBackend()
    sample = ((torch.randn(1, IN_FEATURES),), {})
    backend._graph_spec = make_graph_spec(
        _single_input,
        sample,
        (torch.randn(1, OUT_FEATURES),),
    )
    update_input_spec(backend._graph_spec, ((torch.randn(2, IN_FEATURES),), {}))
    update_input_spec(backend._graph_spec, ((torch.randn(4, IN_FEATURES),), {}))

    profiles = backend.get_profiles(graph_spec=backend._graph_spec, data=[])
    assert len(profiles) == 1

    pr = profiles[0]
    input_name = format_tensor_name("x", "input")
    assert pr[input_name].min == (1, IN_FEATURES)
    assert pr[input_name].opt == (4, IN_FEATURES)
    assert pr[input_name].max == (4, IN_FEATURES)


def test_get_profiles_uses_explicit_shape_range():
    backend = TensorRTBackend()
    sample = ((torch.randn(2, IN_FEATURES),), {})
    backend._graph_spec = make_graph_spec(_single_input, sample, (torch.randn(2, OUT_FEATURES),))
    backend._graph_spec.dynamic_shapes = {
        "x": (BatchDim("batch", min=1, opt=2, max=8), IN_FEATURES),
    }

    profile = backend.get_profiles(graph_spec=backend._graph_spec, data=[])[0]

    input_name = format_tensor_name("x", "input")
    assert profile[input_name].min == (1, IN_FEATURES)
    assert profile[input_name].opt == (2, IN_FEATURES)
    assert profile[input_name].max == (8, IN_FEATURES)


def test_get_profiles_rejects_explicit_shapes_with_user_profiles():
    configured_profile = TensorRTProfile().add_input_shape(
        path="x", min_shape=(1, IN_FEATURES), opt_shape=(2, IN_FEATURES), max_shape=(8, IN_FEATURES)
    )
    backend = TensorRTBackend(TensorRTBackendConfig(profiles=[configured_profile]))
    graph_spec = make_graph_spec(_single_input, ((torch.randn(2, IN_FEATURES),), {}))
    graph_spec.dynamic_shapes = {
        "x": (BatchDim("batch", min=1, opt=2, max=8), IN_FEATURES),
    }

    with pytest.raises(AITuneUserInputError, match="cannot be provided together"):
        backend.get_profiles(graph_spec=graph_spec, data=[])


def test_get_profiles_uses_backend_safe_compound_names():
    def nested_input(inner):
        return inner["a"]

    sample = ((), {"inner": {"a": torch.randn(1, IN_FEATURES)}})
    backend = TensorRTBackend()
    backend._graph_spec = make_graph_spec(nested_input, sample, nested_input(**sample[1]))

    profile = backend.get_profiles(graph_spec=backend._graph_spec, data=[])[0]

    assert profile[format_tensor_name(("inner", "a"), "input")].min == (1, IN_FEATURES)


def test_profiles_eq():
    profile1 = TensorRTProfile().add_input_shape("x", (1, IN_FEATURES), (1, IN_FEATURES), (1, IN_FEATURES))
    profile2 = TensorRTProfile().add_input_shape("x", (1, IN_FEATURES), (1, IN_FEATURES), (1, IN_FEATURES))
    assert profile1 == profile2
    assert hash(profile1) == hash(profile2)

    profile1 = TensorRTProfile().add_input_shape("x", (1, IN_FEATURES), (1, IN_FEATURES), (1, IN_FEATURES))
    profile2 = TensorRTProfile().add_input_shape("y", (1, IN_FEATURES), (1, IN_FEATURES), (1, IN_FEATURES))
    assert profile1 != profile2

    profile1 = TensorRTProfile().add_input_shape("x", (2, IN_FEATURES), (2, IN_FEATURES), (2, IN_FEATURES))
    profile2 = TensorRTProfile().add_input_shape("x", (1, IN_FEATURES), (1, IN_FEATURES), (1, IN_FEATURES))
    assert profile1 != profile2


def test_exception_when_max_num_samples_stored_is_set_to_1():
    with pytest.raises(ValueError):
        TensorRTBackend(config=TensorRTBackendConfig(profiles=ProfileMode.SAMPLES_USED))


@pytest.fixture
def global_config_max_num_samples_all(mocker):
    mock_global_config = mocker.patch("aitune.torch.backend.tensorrt.tensorrt_backend.global_config")
    mock_global_config.max_num_samples_stored = float("inf")
    return mock_global_config


def test_get_profiles_multiple_profiles(global_config_max_num_samples_all):
    """Test the get_profiles method with multiple profiles."""

    backend = TensorRTBackend(config=TensorRTBackendConfig(profiles=ProfileMode.SAMPLES_USED))
    sample = ((torch.randn(1, IN_FEATURES),), {})
    backend._graph_spec = make_graph_spec(
        _single_input,
        sample,
        (torch.randn(1, OUT_FEATURES),),
        batch_size=1,
    )
    update_input_spec(backend._graph_spec, ((torch.randn(8, IN_FEATURES),), {}), batch_size=8)

    profiles = backend.get_profiles(
        graph_spec=backend._graph_spec,
        data=[
            ((torch.randn(1, IN_FEATURES),), {}),
            ((torch.randn(8, IN_FEATURES),), {}),
        ],
    )
    assert len(profiles) == 2

    input_name = format_tensor_name("x", "input")
    pr = profiles[0]
    assert pr[input_name].min == (1, IN_FEATURES)
    assert pr[input_name].opt == (1, IN_FEATURES)
    assert pr[input_name].max == (1, IN_FEATURES)

    pr = profiles[1]
    assert pr[input_name].min == (8, IN_FEATURES)
    assert pr[input_name].opt == (8, IN_FEATURES)
    assert pr[input_name].max == (8, IN_FEATURES)


def test_get_profiles_multiple_profiles_with_kwargs(global_config_max_num_samples_all):
    """Test the _get_profiles method with multiple profiles."""

    samples = [
        ((torch.randn(1, IN_FEATURES),), {"input_tensor": torch.randn(1, IN_FEATURES)}),
        ((torch.randn(8, IN_FEATURES),), {"input_tensor": torch.randn(8, IN_FEATURES)}),
    ]

    backend = TensorRTBackend(config=TensorRTBackendConfig(profiles=ProfileMode.SAMPLES_USED))
    backend._graph_spec = make_graph_spec(
        _profile_inputs,
        samples[0],
        (torch.randn(1, OUT_FEATURES),),
        batch_size=1,
    )
    update_input_spec(backend._graph_spec, samples[1], batch_size=8)

    profiles = backend.get_profiles(graph_spec=backend._graph_spec, data=samples)
    assert len(profiles) == 2

    x_name = format_tensor_name("x", "input")
    input_tensor_name = format_tensor_name("input_tensor", "input")
    pr = profiles[0]
    assert pr[x_name].min == (1, IN_FEATURES)
    assert pr[x_name].opt == (1, IN_FEATURES)
    assert pr[x_name].max == (1, IN_FEATURES)

    assert pr[input_tensor_name].min == (1, IN_FEATURES)
    assert pr[input_tensor_name].opt == (1, IN_FEATURES)
    assert pr[input_tensor_name].max == (1, IN_FEATURES)

    pr = profiles[1]
    assert pr[x_name].min == (8, IN_FEATURES)
    assert pr[x_name].opt == (8, IN_FEATURES)
    assert pr[x_name].max == (8, IN_FEATURES)

    assert pr[input_tensor_name].min == (8, IN_FEATURES)
    assert pr[input_tensor_name].opt == (8, IN_FEATURES)
    assert pr[input_tensor_name].max == (8, IN_FEATURES)


def test_get_profiles_with_user_provided_profiles():
    """Test the _get_profiles method with user provided profiles."""

    config_profiles = [
        TensorRTProfile().add_input_shape(
            path="x", min_shape=(1, IN_FEATURES), opt_shape=(2, IN_FEATURES), max_shape=(4, IN_FEATURES)
        )
    ]
    backend = TensorRTBackend(TensorRTBackendConfig(profiles=config_profiles))
    backend._graph_spec = make_graph_spec(
        _single_input,
        ((torch.randn(1, IN_FEATURES),), {}),
        (torch.randn(1, OUT_FEATURES),),
    )
    profiles = backend.get_profiles(graph_spec=backend._graph_spec, data=[])
    assert len(profiles) == 1
    input_name = format_tensor_name("x", "input")
    assert profiles[0][input_name].min == (1, IN_FEATURES)
    assert profiles[0][input_name].opt == (2, IN_FEATURES)
    assert profiles[0][input_name].max == (4, IN_FEATURES)


def test_get_profiles_rejects_unknown_user_profile_input():
    profile = TensorRTProfile().add_input_shape(
        path="unknown", min_shape=(1, IN_FEATURES), opt_shape=(2, IN_FEATURES), max_shape=(4, IN_FEATURES)
    )
    backend = TensorRTBackend(TensorRTBackendConfig(profiles=[profile]))
    graph_spec = make_graph_spec(
        _single_input,
        ((torch.randn(1, IN_FEATURES),), {}),
        (torch.randn(1, OUT_FEATURES),),
    )

    with pytest.raises(AITuneUserInputError, match="unknown"):
        backend.get_profiles(graph_spec=graph_spec, data=[])


def test_get_profiles_rejects_missing_user_profile_input():
    profile = TensorRTProfile().add_input_shape(
        path="x", min_shape=(1, IN_FEATURES), opt_shape=(2, IN_FEATURES), max_shape=(4, IN_FEATURES)
    )
    backend = TensorRTBackend(TensorRTBackendConfig(profiles=[profile]))
    graph_spec = make_graph_spec(
        _two_inputs,
        ((torch.randn(1, IN_FEATURES), torch.randn(1, IN_FEATURES)), {}),
        (torch.randn(1, OUT_FEATURES),),
    )

    missing_name = format_tensor_name("arg", "input")
    with pytest.raises(AITuneUserInputError, match=missing_name):
        backend.get_profiles(graph_spec=graph_spec, data=[])


def test_save_and_load_trt_optimization_profiles(tmp_path, global_config_max_num_samples_all):
    backend = TensorRTBackend(config=TensorRTBackendConfig(profiles=ProfileMode.SAMPLES_USED))
    backend._graph_spec = make_graph_spec(
        _single_input_with_options,
        ((torch.randn(1, IN_FEATURES),), {}),
        (torch.randn(1, OUT_FEATURES),),
        batch_size=1,
    )
    update_input_spec(backend._graph_spec, ((torch.randn(8, IN_FEATURES),), {}), batch_size=8)
    samples = [
        ((torch.randn(1, IN_FEATURES),), {"input_tensor": torch.randn(1, IN_FEATURES)}),
        ((torch.randn(8, IN_FEATURES),), {"input_tensor": torch.randn(8, IN_FEATURES)}),
    ]
    profiles = backend.get_profiles(graph_spec=backend._graph_spec, data=samples)
    assert len(profiles) == 2

    trt_optimization_profiles_path = backend._save_trt_optimization_profiles(profiles, tmp_path)

    assert trt_optimization_profiles_path.exists()

    loaded_profiles = backend._load_trt_optimization_profiles(trt_optimization_profiles_path)

    assert len(loaded_profiles) == 2
    input_name = format_tensor_name("x", "input")
    assert loaded_profiles[0][input_name].min == (1, IN_FEATURES)
    assert loaded_profiles[0][input_name].opt == (1, IN_FEATURES)
    assert loaded_profiles[0][input_name].max == (1, IN_FEATURES)

    assert loaded_profiles[1][input_name].min == (8, IN_FEATURES)
    assert loaded_profiles[1][input_name].opt == (8, IN_FEATURES)
    assert loaded_profiles[1][input_name].max == (8, IN_FEATURES)


def test_set_optimization_profiles_01(mocker, global_config_max_num_samples_all):
    backend = TensorRTBackend(config=TensorRTBackendConfig(profiles=ProfileMode.SAMPLES_USED))
    backend._trt_optimization_profiles = [
        Profile().add("args_0", (1, IN_FEATURES), (1, IN_FEATURES), (1, IN_FEATURES)),
        Profile().add("args_0", (8, IN_FEATURES), (8, IN_FEATURES), (8, IN_FEATURES)),
    ]
    mock_context = mocker.MagicMock()
    backend._context = mock_context
    backend._cuda_stream = mocker.MagicMock()
    backend._cuda_stream.cuda_stream = "stream"

    backend._set_optimization_profiles({"args_0": torch.randn(1, IN_FEATURES)})
    mock_context.set_optimization_profile_async.assert_called_once_with(0, "stream")

    backend._set_optimization_profiles({"args_0": torch.randn(8, IN_FEATURES)})
    mock_context.set_optimization_profile_async.assert_called_with(1, "stream")


def test_set_optimization_profiles_additional_kwargs(mocker, global_config_max_num_samples_all):
    backend = TensorRTBackend(config=TensorRTBackendConfig(profiles=ProfileMode.SAMPLES_USED))
    backend._trt_optimization_profiles = [
        # first profile with args and kwargs
        Profile().add("args_0", (1, 32), (1, 32), (1, 32)).add("kwargs_input_tensor", (1, 64), (1, 64), (1, 64)),
        # second profile without kwargs_input_tensor
        Profile().add("args_0", (8, 128), (8, 128), (8, 128)),
        # third profile with additional args
        Profile().add("args_0", (8, 256), (8, 256), (8, 256)).add("args_1", (8, 256), (8, 256), (8, 256)),
    ]
    mock_context = mocker.MagicMock()
    backend._context = mock_context
    backend._cuda_stream = mocker.MagicMock()
    backend._cuda_stream.cuda_stream = "stream"

    backend._set_optimization_profiles({
        "args_0": torch.randn(1, 32),
        "kwargs_input_tensor": torch.randn(1, 64),
    })
    mock_context.set_optimization_profile_async.assert_called_once_with(0, "stream")

    backend._set_optimization_profiles({"args_0": torch.randn(8, 128)})
    mock_context.set_optimization_profile_async.assert_called_with(1, "stream")

    backend._set_optimization_profiles({"args_0": torch.randn(8, 256), "args_1": torch.randn(8, 256)})
    mock_context.set_optimization_profile_async.assert_called_with(2, "stream")

    with pytest.raises(RuntimeError):
        backend._set_optimization_profiles({"args_0": torch.randn(8, 128), "kwargs_input_tensor": torch.randn(8, 64)})

    with pytest.raises(RuntimeError):
        backend._set_optimization_profiles({"args_0": torch.randn(8, 256)})


def test_set_optimization_profiles_for_user_provided_profiles(mocker):
    backend = TensorRTBackend(
        config=TensorRTBackendConfig(
            profiles=[
                # we do not use build() in this test, so we provide profiles directly below
            ]
        )
    )
    backend._trt_optimization_profiles = [
        # first profile with args and kwargs
        Profile().add("args_0", (1, 32), (1, 32), (1, 64)).add("kwargs_input_tensor", (1, 32), (1, 64), (1, 64)),
        # second profile without kwargs_input_tensor
        Profile().add("args_0", (8, 128), (8, 128), (8, 256)),
        # third profile with additional args
        Profile().add("args_0", (8, 256), (8, 256), (8, 512)).add("args_1", (8, 256), (8, 256), (8, 512)),
    ]
    mock_context = mocker.MagicMock()
    backend._context = mock_context
    backend._cuda_stream = mocker.MagicMock()
    backend._cuda_stream.cuda_stream = "stream"

    backend._set_optimization_profiles({
        "args_0": torch.randn(1, 48),
        "kwargs_input_tensor": torch.randn(1, 48),
    })
    mock_context.set_optimization_profile_async.assert_called_once_with(0, "stream")

    backend._set_optimization_profiles({"args_0": torch.randn(8, 192)})
    mock_context.set_optimization_profile_async.assert_called_with(1, "stream")

    backend._set_optimization_profiles({"args_0": torch.randn(8, 384), "args_1": torch.randn(8, 384)})
    mock_context.set_optimization_profile_async.assert_called_with(2, "stream")

    with pytest.raises(RuntimeError):
        backend._set_optimization_profiles({"args_0": torch.randn(2, 48), "kwargs_input_tensor": torch.randn(2, 48)})

    with pytest.raises(RuntimeError):
        backend._set_optimization_profiles({"args_0": torch.randn(1, 48), "kwargs_input_tensor": torch.randn(1, 31)})

    with pytest.raises(RuntimeError):
        backend._set_optimization_profiles({"args_0": torch.randn(8, 257)})


def test_tensorrt_backend_config_to_dict_with_profiles():
    config = TensorRTBackendConfig(
        profiles=[
            TensorRTProfile()
            .add_input_shape("x", (1, 32), (1, 32), (1, 32))
            .add_input_shape("input_tensor", (1, 64), (1, 64), (1, 64)),
            TensorRTProfile().add_input_shape("x", (8, 128), (8, 128), (8, 128)),
        ]
    )
    new_config = TensorRTBackendConfig.from_dict(config.to_dict())

    assert len(new_config.profiles) == len(config.profiles)
    x_name = format_tensor_name("x", "input")
    input_tensor_name = format_tensor_name("input_tensor", "input")
    assert x_name in new_config.profiles[0].profile
    assert input_tensor_name in new_config.profiles[0].profile

    assert x_name in new_config.profiles[1].profile
    assert input_tensor_name not in new_config.profiles[1].profile


def test_tensorrt_backend_config_to_dict_with_profiles_mode():
    config = TensorRTBackendConfig(profiles=ProfileMode.SAMPLES_USED)
    new_config = TensorRTBackendConfig.from_dict(config.to_dict())
    assert new_config.profiles == ProfileMode.SAMPLES_USED


# --- TensorRTBackendConfig.from_dict ---


def test_tensorrt_config_from_dict_defaults():
    config = TensorRTBackendConfig.from_dict({})
    assert config == TensorRTBackendConfig()


def test_tensorrt_config_from_dict_profile_mode_single_string():
    config = TensorRTBackendConfig.from_dict({"profiles": "single"})
    assert config.profiles == ProfileMode.SINGLE


def test_tensorrt_config_from_dict_profile_mode_samples_used_string():
    config = TensorRTBackendConfig.from_dict({"profiles": "samples_used"})
    assert config.profiles == ProfileMode.SAMPLES_USED


def test_tensorrt_config_from_dict_custom_fields():
    config = TensorRTBackendConfig.from_dict({"use_dynamo": False, "workspace_size": 1024, "enable_tf32": False})
    assert config.use_dynamo is False
    assert config.workspace_size == 1024
    assert config.enable_tf32 is False


def test_tensorrt_config_from_dict_round_trip():
    original = TensorRTBackendConfig(use_dynamo=False, workspace_size=512)
    restored = TensorRTBackendConfig.from_dict(original.to_dict())
    assert restored.use_dynamo == original.use_dynamo
    assert restored.workspace_size == original.workspace_size
    assert restored.profiles == original.profiles


def test_tensorrt_config_from_dict_with_onnx_autocast_config_instance():
    from aitune.torch.backend.tensorrt.onnx_autocast import ONNXAutoCastConfig

    quant_cfg = ONNXAutoCastConfig(precision="fp16")
    config = TensorRTBackendConfig.from_dict({"quantization_config": quant_cfg})
    assert isinstance(config.quantization_config, ONNXAutoCastConfig)
    assert config.quantization_config.precision == "fp16"


def test_tensorrt_config_from_dict_with_onnx_quantization_config_instance():
    from aitune.torch.backend.tensorrt.onnx_quantization import ONNXQuantizationConfig

    quant_cfg = ONNXQuantizationConfig(precision="int8", calibration_method="max")
    config = TensorRTBackendConfig.from_dict({"quantization_config": quant_cfg})
    assert isinstance(config.quantization_config, ONNXQuantizationConfig)
    assert config.quantization_config.precision == "int8"
    assert config.quantization_config.calibration_method == "max"


def test_tensorrt_config_from_dict_with_torch_quantization_config_instance():
    from aitune.torch.backend.tensorrt.torch_quantization import TorchQuantizationConfig

    quant_cfg = TorchQuantizationConfig(quantization_config="FP8_DEFAULT_CFG")
    config = TensorRTBackendConfig.from_dict({"quantization_config": quant_cfg})
    assert isinstance(config.quantization_config, TorchQuantizationConfig)
    assert config.quantization_config.quantization_config == "FP8_DEFAULT_CFG"


def test_onnx_autocast_config_rejects_invalid_precision():
    from aitune.torch.backend.tensorrt.onnx_autocast import ONNXAutoCastConfig

    with pytest.raises(ValueError, match="Unsupported autocast precision"):
        ONNXAutoCastConfig(precision="invalid")  # pytype: disable=wrong-arg-types


def test_onnx_quantization_config_rejects_invalid_precision():
    from aitune.torch.backend.tensorrt.onnx_quantization import ONNXQuantizationConfig

    with pytest.raises(ValueError, match="Unsupported precision"):
        ONNXQuantizationConfig(precision="invalid")  # pytype: disable=wrong-arg-types


def test_torch_quantization_config_rejects_invalid_config():
    from aitune.torch.backend.tensorrt.torch_quantization import TorchQuantizationConfig

    with pytest.raises(ValueError, match="Unsupported quantization_config"):
        TorchQuantizationConfig(quantization_config="invalid")  # pytype: disable=wrong-arg-types


def test_tensorrt_config_onnx_autocast_round_trip():
    from aitune.torch.backend.tensorrt.onnx_autocast import ONNXAutoCastConfig

    original = TensorRTBackendConfig(quantization_config=ONNXAutoCastConfig(precision="bf16"))
    restored = TensorRTBackendConfig.from_dict(original.to_dict())
    assert isinstance(restored.quantization_config, ONNXAutoCastConfig)
    assert restored.quantization_config.precision == "bf16"


def test_tensorrt_config_onnx_quantization_round_trip():
    from aitune.torch.backend.tensorrt.onnx_quantization import ONNXQuantizationConfig

    original = TensorRTBackendConfig(quantization_config=ONNXQuantizationConfig(precision="fp8"))
    restored = TensorRTBackendConfig.from_dict(original.to_dict())
    assert isinstance(restored.quantization_config, ONNXQuantizationConfig)
    assert restored.quantization_config.precision == "fp8"


def test_tensorrt_config_torch_quantization_round_trip():
    from aitune.torch.backend.tensorrt.torch_quantization import TorchQuantizationConfig

    original = TensorRTBackendConfig(
        quantization_config=TorchQuantizationConfig(quantization_config="INT8_DEFAULT_CFG")
    )
    restored = TensorRTBackendConfig.from_dict(original.to_dict())
    assert isinstance(restored.quantization_config, TorchQuantizationConfig)
    assert restored.quantization_config.quantization_config == "INT8_DEFAULT_CFG"
