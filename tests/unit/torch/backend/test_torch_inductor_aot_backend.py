# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TorchInductorAotBackend."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from aitune.torch.backend.backend import BackendState
from aitune.torch.backend.torch_inductor_aot_backend import (
    TorchInductorAotBackend,
    TorchInductorAotBackendConfig,
)
from aitune.torch.checkpoint.storage_tasks import torch_load_with_custom_types
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.module.sample_metadata import SampleMetadata
from tests.toy_models import ToyTorchModel
from tests.utilities.helpers import requires_cuda

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _empty_graph_spec(input_metadata: SampleMetadata) -> GraphSpec:
    """Wrap input_metadata in a GraphSpec with a dummy output spec."""
    return GraphSpec("test", input_metadata, SampleMetadata.from_inputs((), {}))


@pytest.fixture
def model(torch_device) -> nn.Module:
    return ToyTorchModel().to(torch_device).eval()


@pytest.fixture
def sample_data(torch_device) -> list[Sample]:
    toy = ToyTorchModel()
    return toy.samples(batch_sizes=[1], device=torch_device)


@pytest.fixture
def graph_spec(torch_device) -> GraphSpec:
    toy = ToyTorchModel().to(torch_device)
    return toy.graph_spec(batch_sizes=[1, 2], device=torch_device)


@pytest.fixture
def backend() -> TorchInductorAotBackend:
    return TorchInductorAotBackend()


@pytest.fixture
def mock_aoti(mocker, model):
    """Mock all three external torch AOT calls; runner forwards to the original model."""
    mocker.patch("torch.export.export", return_value=Mock())
    mocker.patch.object(torch._inductor, "aoti_compile_and_package")

    def _fake_runner(*args, **kwargs):
        with torch.no_grad():
            device = next(model.parameters()).device
            args_d = tuple(a.to(device) if isinstance(a, torch.Tensor) else a for a in args)
            kwargs_d = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
            return model(*args_d, **kwargs_d)

    mock_runner = Mock(side_effect=_fake_runner)
    mocker.patch.object(torch._inductor, "aoti_load_package", return_value=mock_runner)
    return mock_runner


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_config_key_is_stable():
    config = TorchInductorAotBackendConfig()
    assert config.key() == config.key()


def test_config_key_differs_for_different_options():
    config_a = TorchInductorAotBackendConfig()
    config_b = TorchInductorAotBackendConfig(inductor_configs={"max_autotune": True})
    assert config_a.key() != config_b.key()


def test_config_describe_default():
    config = TorchInductorAotBackendConfig()
    assert config.describe() == ""


def test_config_describe_with_inductor_configs():
    config = TorchInductorAotBackendConfig(inductor_configs={"max_autotune": True})
    description = config.describe()
    assert "inductor_configs" in description


def test_config_roundtrip():
    config = TorchInductorAotBackendConfig(inductor_configs={"max_autotune": True})
    assert TorchInductorAotBackendConfig.from_dict(config.to_dict()).inductor_configs == config.inductor_configs


# ---------------------------------------------------------------------------
# _build_dynamic_shapes tests (no GPU required)
# ---------------------------------------------------------------------------


def test_build_dynamic_shapes_static(backend):
    """Single sample → no shape updates → all static dims → None."""
    args = (torch.randn(2, 32),)
    meta = SampleMetadata.from_inputs(args, {}, batch_size=2)
    result = backend._build_dynamic_shapes(args, {}, _empty_graph_spec(meta))
    assert result is None


def test_build_dynamic_shapes_batch_axis(backend):
    """Two samples with different batch sizes → axis 0 detected as batch."""
    args1 = (torch.randn(1, 32),)
    args2 = (torch.randn(2, 32),)
    meta = SampleMetadata.from_inputs(args1, {}, batch_size=1)
    meta.update_shapes_seen(SampleMetadata.from_inputs(args2, {}, batch_size=2))

    result = backend._build_dynamic_shapes(args1, {}, _empty_graph_spec(meta))

    assert result is not None
    assert len(result) == 1
    assert isinstance(result[0], dict)
    assert 0 in result[0], "axis 0 should be marked as batch dynamic"


def test_build_dynamic_shapes_dynamic_axis(backend):
    """Same batch size but different sequence length → axis 1 detected as dynamic."""
    args1 = (torch.randn(2, 5),)
    args2 = (torch.randn(2, 7),)
    meta = SampleMetadata.from_inputs(args1, {}, batch_size=2)
    meta.update_shapes_seen(SampleMetadata.from_inputs(args2, {}, batch_size=2))

    result = backend._build_dynamic_shapes(args1, {}, _empty_graph_spec(meta))

    assert result is not None
    assert isinstance(result[0], dict)
    assert 1 in result[0], "axis 1 should be marked as dynamic (non-batch)"


def test_build_dynamic_shapes_mixed_batch_and_dynamic(backend):
    """Different batch size AND non-proportional axis → both batch and dim axes present."""
    args1 = (torch.randn(1, 5),)  # bs=1, seq=5
    args2 = (torch.randn(2, 7),)  # bs=2, seq=7 (not proportional → dim1)
    meta = SampleMetadata.from_inputs(args1, {}, batch_size=1)
    meta.update_shapes_seen(SampleMetadata.from_inputs(args2, {}, batch_size=2))

    result = backend._build_dynamic_shapes(args1, {}, _empty_graph_spec(meta))

    assert result is not None
    assert 0 in result[0], "axis 0 should be batch"
    assert 1 in result[0], "axis 1 should be dynamic"


def test_build_dynamic_shapes_batch_multiplier(backend):
    """Axis 0 = 2 * batch_size → derived dim should be applied."""
    args1 = (torch.randn(2, 10),)  # bs=1, axis_0 = 2*bs
    args2 = (torch.randn(4, 10),)  # bs=2, axis_0 = 2*bs
    meta = SampleMetadata.from_inputs(args1, {}, batch_size=1)
    meta.update_shapes_seen(SampleMetadata.from_inputs(args2, {}, batch_size=2))

    result = backend._build_dynamic_shapes(args1, {}, _empty_graph_spec(meta))

    assert result is not None
    assert 0 in result[0], "axis 0 should carry a derived batch dim"


def test_build_dynamic_shapes_shared_dim_across_tensors(backend):
    """Two args tensors sharing the same dynamic dim name get the same Dim object."""
    args1 = (torch.randn(2, 5), torch.randn(2, 5))
    args2 = (torch.randn(2, 7), torch.randn(2, 7))
    meta = SampleMetadata.from_inputs(args1, {}, batch_size=2)
    meta.update_shapes_seen(SampleMetadata.from_inputs(args2, {}, batch_size=2))

    result = backend._build_dynamic_shapes(args1, {}, _empty_graph_spec(meta))

    assert result is not None
    assert result[0][1] is result[1][1], "shared dim name must map to the same Dim instance"


def test_build_dynamic_shapes_kwargs_not_covered(backend):
    """Tensors only in kwargs → no args batch axis → returns None."""
    arg = torch.randn(2, 32)
    kw1 = {"x": torch.randn(1, 10)}
    kw2 = {"x": torch.randn(2, 10)}
    # kwargs tensor changes batch; args tensor stays static
    meta = SampleMetadata.from_inputs((arg,), kw1, batch_size=1)
    meta.update_shapes_seen(SampleMetadata.from_inputs((arg,), kw2, batch_size=2))

    result = backend._build_dynamic_shapes((arg,), kw1, _empty_graph_spec(meta))
    assert result is None


# ---------------------------------------------------------------------------
# Build / infer / state tests (GPU)
# ---------------------------------------------------------------------------


@requires_cuda
def test_build_returns_active_backend(mock_aoti, backend, model, graph_spec, sample_data, torch_device, tmp_path):
    built = backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)
    assert built is backend
    assert backend.is_active
    assert backend._compiled_model_path == tmp_path / "model.pt2"


@requires_cuda
def test_build_calls_export_and_compile(
    mock_aoti, mocker, backend, model, graph_spec, sample_data, torch_device, tmp_path
):
    export_mock = mocker.patch("torch.export.export", return_value=Mock())
    compile_mock = mocker.patch.object(torch._inductor, "aoti_compile_and_package")

    backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)

    export_mock.assert_called_once()
    compile_mock.assert_called_once()
    _, call_kwargs = compile_mock.call_args
    assert call_kwargs["package_path"] == str(tmp_path / "model.pt2")


@requires_cuda
def test_build_passes_dynamic_shapes_when_batch_detected(
    mock_aoti, mocker, model, graph_spec, sample_data, torch_device, tmp_path
):
    export_mock = mocker.patch("torch.export.export", return_value=Mock())
    mocker.patch.object(torch._inductor, "aoti_compile_and_package")

    TorchInductorAotBackend().build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)

    _, call_kwargs = export_mock.call_args
    assert call_kwargs.get("dynamic_shapes") is not None


@requires_cuda
def test_build_static_graph_no_dynamic_shapes(mocker, model, torch_device, tmp_path):
    """Single-sample graph_spec has no dynamic axes → dynamic_shapes=None in export call."""
    toy = ToyTorchModel().to(torch_device)
    gs = toy.graph_spec(batch_sizes=[2], device=torch_device)  # only one batch size → static
    samples = toy.samples(batch_sizes=[2], device=torch_device)

    export_mock = mocker.patch("torch.export.export", return_value=Mock())
    mocker.patch.object(torch._inductor, "aoti_compile_and_package")
    mocker.patch.object(torch._inductor, "aoti_load_package", return_value=Mock())

    TorchInductorAotBackend().build(model, gs, samples, device=torch_device, cache_dir=tmp_path)

    _, call_kwargs = export_mock.call_args
    assert call_kwargs.get("dynamic_shapes") is None


@requires_cuda
def test_infer_calls_runner(mock_aoti, backend, model, graph_spec, sample_data, torch_device, tmp_path):
    backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)
    args, kwargs = sample_data[0]
    output = backend.infer(*args, **kwargs)
    mock_aoti.assert_called_once_with(*args, **kwargs)
    assert output is not None


@requires_cuda
def test_deactivate_clears_runner(mock_aoti, backend, model, graph_spec, sample_data, torch_device, tmp_path):
    backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)
    backend.deactivate()
    assert backend._runner is None
    assert backend.state == BackendState.INACTIVE


@requires_cuda
def test_activate_reloads_runner(mock_aoti, backend, model, graph_spec, sample_data, torch_device, tmp_path):
    load_mock = torch._inductor.aoti_load_package  # already patched by mock_aoti
    backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)
    backend.deactivate()

    initial_call_count = load_mock.call_count
    backend.activate()

    assert backend._runner is not None
    assert load_mock.call_count == initial_call_count + 1


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


def test_to_dict_before_build_raises():
    with pytest.raises(RuntimeError, match="build"):
        TorchInductorAotBackend().to_dict()


@requires_cuda
def test_to_dict_contains_required_keys(mock_aoti, backend, model, graph_spec, sample_data, torch_device, tmp_path):
    backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)
    state = backend.to_dict()
    assert state[TorchInductorAotBackend.STATE_TYPE] == "TorchInductorAotBackend"
    assert isinstance(state[TorchInductorAotBackend.STATE_COMPILED_MODEL_PATH], Path)
    assert state[TorchInductorAotBackend.STATE_DEVICE] == torch_device


def test_from_dict_restores_state(tmp_path, torch_device):
    compiled_path = tmp_path / "model.pt2"
    state = {
        TorchInductorAotBackend.STATE_TYPE: "TorchInductorAotBackend",
        TorchInductorAotBackend.STATE_COMPILED_MODEL_PATH: compiled_path,
        TorchInductorAotBackend.STATE_DEVICE: torch_device,
    }
    restored = TorchInductorAotBackend.from_dict(None, state)
    assert restored._compiled_model_path == compiled_path
    assert restored._device == torch_device
    assert restored.state == BackendState.CHECKPOINT_LOADED


def test_from_dict_wrong_type_raises(tmp_path, torch_device):
    state = {TorchInductorAotBackend.STATE_TYPE: "WrongBackend"}
    with pytest.raises(ValueError, match="Invalid state_dict type"):
        TorchInductorAotBackend.from_dict(None, state)


@requires_cuda
def test_serialization_round_trip(torch_device, tmp_path):
    """Full build → save → load → infer round-trip with a real model."""
    toy = ToyTorchModel().to(torch_device).eval()
    samples = toy.samples(batch_sizes=[1, 2], device=torch_device)
    gs = toy.graph_spec(batch_sizes=[1, 2], device=torch_device)

    backend = TorchInductorAotBackend()
    backend.build(toy, gs, samples, device=torch_device, cache_dir=tmp_path)

    state = backend.to_dict()
    torch.save(state, tmp_path / "state.pth")
    loaded_state = torch_load_with_custom_types(tmp_path / "state.pth")
    loaded = TorchInductorAotBackend.from_dict(None, loaded_state)
    loaded.activate()

    args, kwargs = samples[0]
    torch.testing.assert_close(backend.infer(*args, **kwargs), loaded.infer(*args, **kwargs))
