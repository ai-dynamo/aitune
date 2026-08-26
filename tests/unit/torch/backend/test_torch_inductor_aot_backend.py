# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TorchInductorAotBackend."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from aitune.torch.backend import ArtifactPath
from aitune.torch.backend.backend import BackendState
from aitune.torch.backend.torch_inductor_aot_backend import (
    TorchInductorAotBackend,
    TorchInductorAotBackendConfig,
)
from aitune.torch.checkpoint.storage_tasks import torch_load_with_custom_types
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import Sample
from tests.toy_models import ToyTorchModel
from tests.utilities.helpers import requires_cuda

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


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


def _fake_aoti_compile(*args, **kwargs):
    Path(kwargs["package_path"]).write_bytes(b"fake")


@pytest.fixture
def mock_aoti(mocker, model):
    """Mock all three external torch AOT calls; runner forwards to the original model."""
    mocker.patch("torch.export.export", return_value=Mock())
    mocker.patch.object(torch._inductor, "aoti_compile_and_package", side_effect=_fake_aoti_compile)

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
# Build / infer / state tests (GPU)
# ---------------------------------------------------------------------------


@requires_cuda
def test_build_returns_active_backend(mock_aoti, backend, model, graph_spec, sample_data, torch_device, tmp_path):
    built = backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)
    assert built is backend
    assert backend.is_active
    assert backend._compiled_model_artifact == ArtifactPath(tmp_path, Path("model.pt2"))


@requires_cuda
def test_build_calls_export_and_compile(
    mock_aoti, mocker, backend, model, graph_spec, sample_data, torch_device, tmp_path
):
    export_mock = mocker.patch("torch.export.export", return_value=Mock())
    compile_mock = mocker.patch.object(torch._inductor, "aoti_compile_and_package", side_effect=_fake_aoti_compile)

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
    mocker.patch.object(torch._inductor, "aoti_compile_and_package", side_effect=_fake_aoti_compile)

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
    mocker.patch.object(torch._inductor, "aoti_compile_and_package", side_effect=_fake_aoti_compile)
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
    assert state[TorchInductorAotBackend.STATE_COMPILED_MODEL_PATH] == ArtifactPath(tmp_path, Path("model.pt2"))
    assert state[TorchInductorAotBackend.STATE_DEVICE] == torch_device


@requires_cuda
def test_from_dict_restores_state(tmp_path, torch_device):
    compiled_artifact = ArtifactPath(tmp_path, "model.pt2")
    state = {
        TorchInductorAotBackend.STATE_TYPE: "TorchInductorAotBackend",
        TorchInductorAotBackend.STATE_COMPILED_MODEL_PATH: compiled_artifact,
        TorchInductorAotBackend.STATE_DEVICE: torch_device,
    }
    restored = TorchInductorAotBackend.from_dict(None, state)
    assert restored._compiled_model_artifact == compiled_artifact
    assert restored._device == torch_device
    assert restored.state == BackendState.CHECKPOINT_LOADED


@requires_cuda
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


# --- TorchInductorAotBackendConfig.from_dict ---


def test_inductor_aot_config_from_dict_defaults():
    config = TorchInductorAotBackendConfig.from_dict({})
    assert config == TorchInductorAotBackendConfig()


def test_inductor_aot_config_from_dict_custom_fields():
    config = TorchInductorAotBackendConfig.from_dict({"inductor_configs": {"max_autotune": True}})
    assert config.inductor_configs == {"max_autotune": True}


def test_inductor_aot_config_from_dict_round_trip():
    original = TorchInductorAotBackendConfig(inductor_configs={"coordinate_descent_tuning": True})
    restored = TorchInductorAotBackendConfig.from_dict(original.to_dict())
    assert restored == original
