# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ONNXRuntimeBackend."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from aitune.torch.backend.backend import BackendState
from aitune.torch.backend.onnx_runtime_backend import (
    ONNXExecutionProvider,
    ONNXRuntimeBackend,
    ONNXRuntimeBackendConfig,
)
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from tests.toy_models import ToyTorchModel
from tests.utilities.helpers import requires_cuda

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(name: str, min_shape: tuple, max_shape: tuple):
    """Create a minimal mock TensorSpec."""
    ts = Mock()
    ts.name = name
    ts.min_shape = min_shape
    ts.max_shape = max_shape
    return ts


def _gs(input_specs=(), output_specs=()):
    """Create a mock GraphSpec with the given tensor specs."""
    gs = Mock()
    gs.input_spec.tensor_specs = list(input_specs)
    gs.output_spec.tensor_specs = list(output_specs)
    return gs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model(torch_device) -> nn.Module:
    return ToyTorchModel().to(torch_device).eval()


@pytest.fixture
def sample_data(torch_device) -> list[Sample]:
    return ToyTorchModel().samples(batch_sizes=[1], device=torch_device)


@pytest.fixture
def graph_spec(torch_device) -> GraphSpec:
    return ToyTorchModel().to(torch_device).graph_spec(batch_sizes=[1, 2], device=torch_device)


@pytest.fixture
def backend() -> ONNXRuntimeBackend:
    return ONNXRuntimeBackend()


@pytest.fixture
def mock_onnx(mocker, model):
    """Mock torch.onnx.export, onnx.checker.check_model, and onnxruntime.InferenceSession.

    memcpy_to_torch is patched to return a fixed tensor so _collect_outputs works
    in CPU-only test environments without a real CUDA device or libcudart.
    """
    mocker.patch("torch.onnx.export")
    mocker.patch("onnx.checker.check_model")
    mocker.patch(
        "aitune.torch.backend.onnx_runtime_backend.memcpy_to_torch",
        return_value=torch.zeros(1, 5),
    )

    def _make_session(path, providers=None):
        mock_sess = Mock()
        mock_inp = Mock()
        mock_inp.name = "args_0"
        mock_sess.get_inputs.return_value = [mock_inp]
        mock_out = Mock()
        mock_out.name = "outputs_0"
        mock_sess.get_outputs.return_value = [mock_out]
        mock_ort_val = Mock()
        mock_ort_val.data_ptr.return_value = 0
        mock_ort_val.shape.return_value = [1, 5]
        mock_ort_val.data_type.return_value = "tensor(float)"
        mock_io = Mock()
        mock_io.get_outputs.return_value = [mock_ort_val]
        mock_sess.io_binding.return_value = mock_io
        return mock_sess

    mocker.patch("onnxruntime.InferenceSession", side_effect=_make_session)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


@requires_cuda
def test_config_key_is_stable():
    config = ONNXRuntimeBackendConfig()
    assert config.key() == config.key()


@requires_cuda
def test_config_key_differs_for_use_dynamo():
    assert ONNXRuntimeBackendConfig(use_dynamo=False).key() != ONNXRuntimeBackendConfig(use_dynamo=True).key()


@requires_cuda
def test_config_describe_default():
    assert ONNXRuntimeBackendConfig().describe() == ""


@requires_cuda
def test_config_roundtrip():
    config = ONNXRuntimeBackendConfig(use_dynamo=True)
    restored = ONNXRuntimeBackendConfig.from_dict(config.to_dict())
    assert restored.use_dynamo == config.use_dynamo


# ---------------------------------------------------------------------------
# _get_execution_providers tests
# ---------------------------------------------------------------------------


@requires_cuda
def test_get_execution_providers_cuda(backend):
    backend._device = torch.device("cuda", 0)
    assert backend._get_execution_providers() == [("CUDAExecutionProvider", {"device_id": 0})]


@requires_cuda
def test_get_execution_providers_cuda_device_index_propagated():
    """CUDA device index is forwarded to the CUDAExecutionProvider options."""
    b = ONNXRuntimeBackend()
    b._device = torch.device("cuda", 2)
    providers = b._get_execution_providers()
    assert providers == [("CUDAExecutionProvider", {"device_id": 2})]


@requires_cuda
def test_get_execution_providers_tensorrt_returns_trt_and_cuda():
    """TENSORRT provider returns TensorrtExecutionProvider + CUDAExecutionProvider fallback."""
    config = ONNXRuntimeBackendConfig(execution_provider=ONNXExecutionProvider.TENSORRT)
    b = ONNXRuntimeBackend(config=config)
    b._device = torch.device("cuda", 0)
    providers = b._get_execution_providers()
    assert providers[0] == "TensorrtExecutionProvider"
    assert providers[1] == ("CUDAExecutionProvider", {"device_id": 0})


# ---------------------------------------------------------------------------
# TensorRT Execution Provider tests
# ---------------------------------------------------------------------------


@requires_cuda
def test_get_execution_providers_tensorrt_ep():
    """TENSORRT provider returns TensorrtExecutionProvider + CUDA fallback."""
    b = ONNXRuntimeBackend(ONNXRuntimeBackendConfig(execution_provider=ONNXExecutionProvider.TENSORRT))
    b._device = torch.device("cuda", 0)
    providers = b._get_execution_providers()
    assert providers[0] == "TensorrtExecutionProvider"
    assert providers[1] == ("CUDAExecutionProvider", {"device_id": 0})


@requires_cuda
def test_config_key_differs_for_tensorrt_ep():
    """TRT EP config produces a different key than the default CUDA EP config."""
    default = ONNXRuntimeBackendConfig()
    trt_ep = ONNXRuntimeBackendConfig(execution_provider=ONNXExecutionProvider.TENSORRT)
    assert default.key() != trt_ep.key()


@requires_cuda
def test_config_describe_includes_execution_provider():
    """TRT EP shows up in the config description."""
    config = ONNXRuntimeBackendConfig(execution_provider=ONNXExecutionProvider.TENSORRT)
    assert "execution_provider" in config.describe()


@requires_cuda
def test_build_with_tensorrt_ep_creates_session_with_correct_providers(
    mock_onnx, mocker, model, graph_spec, sample_data, torch_device, tmp_path
):
    """InferenceSession must be created with TensorrtExecutionProvider as first provider."""
    import onnxruntime

    session_constructor = onnxruntime.InferenceSession  # already patched by mock_onnx
    ONNXRuntimeBackend(ONNXRuntimeBackendConfig(execution_provider=ONNXExecutionProvider.TENSORRT)).build(
        model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path
    )
    _, call_kwargs = session_constructor.call_args
    providers = call_kwargs.get("providers")
    assert providers[0] == "TensorrtExecutionProvider"


@requires_cuda
def test_build_with_tensorrt_ep_returns_active_backend(
    mock_onnx, model, graph_spec, sample_data, torch_device, tmp_path
):
    """Backend built with TRT EP config is active and ready for inference."""
    backend = ONNXRuntimeBackend(ONNXRuntimeBackendConfig(execution_provider=ONNXExecutionProvider.TENSORRT))
    backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)
    assert backend.is_active


# ---------------------------------------------------------------------------
# Build / infer / state tests
# ---------------------------------------------------------------------------


@requires_cuda
def test_build_returns_active_backend(mock_onnx, backend, model, graph_spec, sample_data, torch_device, tmp_path):
    built = backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)
    assert built is backend
    assert backend.is_active
    assert backend._onnx_model_path == tmp_path / "model_raw.onnx"


@requires_cuda
def test_build_default_calls_onnx_export_with_dynamo_true(
    mock_onnx, mocker, backend, model, graph_spec, sample_data, torch_device, tmp_path
):
    """Default config uses dynamo=True export."""
    export_mock = mocker.patch("torch.onnx.export")
    backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)
    export_mock.assert_called_once()
    _, call_kwargs = export_mock.call_args
    assert call_kwargs.get("dynamo") is True


@requires_cuda
def test_build_trace_calls_onnx_export_with_dynamo_false(
    mock_onnx, mocker, model, graph_spec, sample_data, torch_device, tmp_path
):
    """Explicit use_dynamo=False uses the trace-based exporter."""
    export_mock = mocker.patch("torch.onnx.export")
    ONNXRuntimeBackend(ONNXRuntimeBackendConfig(use_dynamo=False)).build(
        model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path
    )
    export_mock.assert_called_once()
    _, call_kwargs = export_mock.call_args
    assert call_kwargs.get("dynamo") is False


@requires_cuda
def test_build_dynamo_passes_dynamic_shapes_for_batch_graph(
    mock_onnx, mocker, model, graph_spec, sample_data, torch_device, tmp_path
):
    """Default config with a dynamic batch graph_spec → dynamic_shapes forwarded to torch.onnx.export."""
    export_mock = mocker.patch("torch.onnx.export")
    ONNXRuntimeBackend().build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)
    _, call_kwargs = export_mock.call_args
    assert call_kwargs.get("dynamic_shapes") is not None


@requires_cuda
def test_build_dynamo_static_graph_no_dynamic_shapes(mocker, torch_device, tmp_path):
    """Single batch size (static graph) → dynamic_shapes=None in export call."""
    mocker.patch("onnxruntime.InferenceSession", return_value=Mock())
    mocker.patch("onnx.checker.check_model")
    export_mock = mocker.patch("torch.onnx.export")

    toy = ToyTorchModel().to(torch_device)
    gs = toy.graph_spec(batch_sizes=[2], device=torch_device)
    samples = toy.samples(batch_sizes=[2], device=torch_device)

    ONNXRuntimeBackend().build(toy, gs, samples, device=torch_device, cache_dir=tmp_path)
    _, call_kwargs = export_mock.call_args
    assert call_kwargs.get("dynamic_shapes") == [{}]


@requires_cuda
def test_infer_calls_session_and_returns_tensor(
    mock_onnx, backend, model, graph_spec, sample_data, torch_device, tmp_path
):
    backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)
    args, kwargs = sample_data[0]
    output = backend.infer(*args, **kwargs)
    backend._session.run_with_iobinding.assert_called_once()
    assert isinstance(output, torch.Tensor)


@requires_cuda
def test_infer_binds_cuda_inputs_via_pointer(
    mock_onnx, backend, model, graph_spec, sample_data, torch_device, tmp_path
):
    """CUDA tensors must go through the zero-copy memory-pointer path, never bind_cpu_input."""
    backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)
    args, kwargs = sample_data[0]
    backend.infer(*args, **kwargs)

    io_binding = backend._session.io_binding.return_value
    io_binding.bind_input.assert_called()
    assert io_binding.bind_input.call_args_list[0].kwargs["device_type"] == "cuda"
    io_binding.bind_cpu_input.assert_not_called()


@requires_cuda
def test_infer_binds_outputs_on_cuda(mock_onnx, backend, model, graph_spec, sample_data, torch_device, tmp_path):
    """Outputs are bound to the CUDA device; ORT handles allocation and shape resolution."""
    backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)
    args, kwargs = sample_data[0]
    backend.infer(*args, **kwargs)

    io_binding = backend._session.io_binding.return_value
    assert io_binding.bind_output.called
    for call in io_binding.bind_output.call_args_list:
        assert call.args[1] == "cuda"


@requires_cuda
def test_deactivate_clears_session(mock_onnx, backend, model, graph_spec, sample_data, torch_device, tmp_path):
    backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)
    backend.deactivate()
    assert backend._session is None
    assert backend.state == BackendState.INACTIVE


@requires_cuda
def test_activate_reloads_session(mock_onnx, backend, model, graph_spec, sample_data, torch_device, tmp_path):
    import onnxruntime

    session_constructor = onnxruntime.InferenceSession  # already patched by mock_onnx
    backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)
    backend.deactivate()

    initial_count = session_constructor.call_count
    backend.activate()

    assert backend._session is not None
    assert session_constructor.call_count == initial_count + 1


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


@requires_cuda
def test_to_dict_before_build_raises():
    with pytest.raises(RuntimeError, match="build"):
        ONNXRuntimeBackend().to_dict()


@requires_cuda
def test_to_dict_contains_required_keys(mock_onnx, backend, model, graph_spec, sample_data, torch_device, tmp_path):
    backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)
    state = backend.to_dict()
    assert state[ONNXRuntimeBackend.STATE_TYPE] == "ONNXRuntimeBackend"
    assert isinstance(state[ONNXRuntimeBackend.STATE_ONNX_MODEL_PATH], Path)
    assert state[ONNXRuntimeBackend.STATE_DEVICE] == torch_device
    assert ONNXRuntimeBackend.STATE_GRAPH_SPEC in state
    assert ONNXRuntimeBackend.STATE_OUTPUT_OBJECT in state


@requires_cuda
def test_from_dict_restores_state(tmp_path, torch_device):
    toy = ToyTorchModel().to(torch_device)
    real_graph_spec = toy.graph_spec(batch_sizes=[1, 2], device=torch_device)
    dummy_output = torch.zeros(1, 5)

    config = ONNXRuntimeBackendConfig()
    onnx_path = tmp_path / "model_raw.onnx"
    state = {
        ONNXRuntimeBackend.STATE_TYPE: "ONNXRuntimeBackend",
        ONNXRuntimeBackend.STATE_ONNX_MODEL_PATH: onnx_path,
        ONNXRuntimeBackend.STATE_DEVICE: torch_device,
        ONNXRuntimeBackend.STATE_CONFIG: config.to_dict(),
        ONNXRuntimeBackend.STATE_GRAPH_SPEC: real_graph_spec.to_dict(),
        ONNXRuntimeBackend.STATE_OUTPUT_OBJECT: dummy_output,
    }
    restored = ONNXRuntimeBackend.from_dict(None, state)
    assert restored._onnx_model_path == onnx_path
    assert restored._device == torch_device
    assert restored._graph_spec is not None
    assert restored._output_object is not None
    assert restored.state == BackendState.CHECKPOINT_LOADED


@requires_cuda
def test_from_dict_wrong_type_raises():
    with pytest.raises(ValueError, match="Invalid state_dict type"):
        ONNXRuntimeBackend.from_dict(None, {ONNXRuntimeBackend.STATE_TYPE: "WrongBackend"})


# --- ONNXRuntimeBackendConfig.from_dict ---


def test_onnx_config_from_dict_defaults():
    config = ONNXRuntimeBackendConfig.from_dict({})
    assert config == ONNXRuntimeBackendConfig()


def test_onnx_config_from_dict_string_execution_provider_cuda():
    config = ONNXRuntimeBackendConfig.from_dict({"execution_provider": "cuda"})
    assert config.execution_provider == ONNXExecutionProvider.CUDA


def test_onnx_config_from_dict_string_execution_provider_tensorrt():
    config = ONNXRuntimeBackendConfig.from_dict({"execution_provider": "tensorrt"})
    assert config.execution_provider == ONNXExecutionProvider.TENSORRT


def test_onnx_config_from_dict_none_execution_provider():
    config = ONNXRuntimeBackendConfig.from_dict({"execution_provider": None})
    assert config.execution_provider is None


def test_onnx_config_rejects_invalid_execution_provider():
    with pytest.raises(ValueError, match="Invalid execution_provider"):
        ONNXRuntimeBackendConfig(execution_provider="invalid")  # pytype: disable=wrong-arg-types


def test_onnx_config_from_dict_round_trip():
    original = ONNXRuntimeBackendConfig(execution_provider=ONNXExecutionProvider.CUDA, use_dynamo=False)
    restored = ONNXRuntimeBackendConfig.from_dict(original.to_dict())
    assert restored == original
