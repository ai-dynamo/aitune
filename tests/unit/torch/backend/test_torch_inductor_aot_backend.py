# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TorchInductorAotBackend."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from aitune.records import BoundedTensorSpec, DeploymentArtifact, DType, RuntimeConfig
from aitune.torch.backend import ArtifactPath
from aitune.torch.backend.backend import BackendState
from aitune.torch.backend.torch_inductor_aot_backend import (
    TorchInductorAotBackend,
    TorchInductorAotBackendConfig,
)
from aitune.torch.checkpoint.storage_tasks import torch_load_with_custom_types
from aitune.torch.module.forward_signature import ForwardSignature
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.module.sample_store import Sample, SampleStore
from aitune.torch.utils.pt2_artifact import PT2CallContract
from tests.toy_models import ToyTorchModel
from tests.utilities.helpers import requires_cuda

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model(torch_device) -> nn.Module:
    return ToyTorchModel().to(torch_device).eval()


@pytest.fixture
def sample_data(torch_device, tmp_path) -> SampleStore:
    return ToyTorchModel().sample_store(tmp_path, batch_sizes=[1], device=torch_device)


@pytest.fixture
def graph_spec(torch_device) -> GraphSpec:
    toy = ToyTorchModel().to(torch_device)
    return toy.graph_spec(batch_sizes=[1, 2], device=torch_device)


@pytest.fixture
def backend() -> TorchInductorAotBackend:
    return TorchInductorAotBackend()


def _fake_aoti_compile(*args, **kwargs):
    Path(kwargs["package_path"]).write_bytes(b"fake")


def _graph_spec_for(model: nn.Module, samples: list[Sample]) -> GraphSpec:
    """Record one graph specification from already batched samples."""
    forward_signature = ForwardSignature.from_callable(model.forward)
    graph_spec = None
    for args, kwargs in samples:
        normalized = forward_signature.normalize(args, kwargs)
        with torch.no_grad():
            output = model(*args, **kwargs)
        inputs = SampleMetadata.from_inputs(normalized.arguments, batch_size=args[0].shape[0])
        outputs = SampleMetadata.from_outputs(output, batch_size=args[0].shape[0])
        if graph_spec is None:
            graph_spec = GraphSpec("structured", inputs, outputs, forward_signature)
        else:
            graph_spec.update_shapes_seen(inputs, outputs)
    if graph_spec is None:
        raise ValueError("At least one sample is required")
    return graph_spec


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
def test_artifact_after_deactivation_exposes_pt2_ordinal_tensor_interface(
    mock_aoti, backend, model, graph_spec, sample_data, torch_device, tmp_path
):
    backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)
    backend.deactivate()

    artifact = backend.artifact()

    assert isinstance(artifact, DeploymentArtifact)
    assert artifact.inputs == (
        BoundedTensorSpec(
            name="INPUT__0",
            dtype=DType.FLOAT32,
            min_shape=(1, 32),
            max_shape=(2, 32),
            batch_axis=0,
        ),
    )
    assert artifact.outputs == (
        BoundedTensorSpec(
            name="OUTPUT__0",
            dtype=DType.FLOAT32,
            min_shape=(1, 5),
            max_shape=(2, 5),
            batch_axis=0,
        ),
    )
    assert artifact.model.format == "pt2"
    assert artifact.model.path == tmp_path / "model.pt2"
    assert artifact.model.additional_files == ()
    assert artifact.model.files == (artifact.model.path,)
    assert artifact.model.metadata == {"structured_call": False}
    assert artifact.runtime == RuntimeConfig(name="aotinductor")
    assert tuple(sample.name for sample in artifact.sample_inputs) == ("INPUT__0",)
    destination = tmp_path / "exported" / "renamed.pt2"
    assert artifact.model.export_files(destination) == destination
    assert destination.read_bytes() == b"fake"


def test_artifact_before_build_raises(backend):
    with pytest.raises(RuntimeError, match="requires a compiled package"):
        backend.artifact()


@requires_cuda
def test_artifact_metadata_failure_does_not_fail_backend_build(
    mock_aoti, mocker, backend, model, graph_spec, sample_data, torch_device, tmp_path
):
    create_artifact = mocker.patch.object(
        backend,
        "_create_artifact",
        side_effect=ValueError("unsupported interface"),
    )

    backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)
    assert backend.is_active
    create_artifact.assert_not_called()
    backend.deactivate()
    backend.activate()
    create_artifact.assert_not_called()

    with pytest.raises(RuntimeError, match="unsupported interface"):
        backend.artifact()
    create_artifact.assert_called_once()
    assert backend.is_active


def test_build_exposes_structured_pt2_call_in_pytree_order(mocker, tmp_path):
    class StructuredModel(nn.Module):
        def forward(self, hidden, options, *, bias):
            total = hidden + options["mask"] + options["residuals"][0] + bias
            return {"total": total, "parts": (hidden - bias, options["mask"] * options["residuals"][0])}

    model = StructuredModel().eval()
    samples = []
    for batch_size in (1, 2):
        tensors = [torch.full((batch_size, 4), value) for value in (1.0, 2.0, 3.0, 4.0)]
        samples.append(((tensors[0], {"mask": tensors[1], "residuals": [tensors[2]]}), {"bias": tensors[3]}))
    graph_spec = _graph_spec_for(model, samples)
    mocker.patch("torch.export.export", return_value=Mock())
    mocker.patch.object(torch._inductor, "aoti_compile_and_package", side_effect=_fake_aoti_compile)
    mocker.patch.object(torch._inductor, "aoti_load_package", return_value=Mock())

    backend = TorchInductorAotBackend().build(
        model,
        graph_spec,
        samples,
        device=torch.device("cpu"),
        cache_dir=tmp_path,
    )

    artifact = backend.artifact()
    assert artifact.input_names == ("INPUT__0", "INPUT__1", "INPUT__2", "INPUT__3")
    assert artifact.output_names == ("OUTPUT__0", "OUTPUT__1", "OUTPUT__2")
    assert artifact.model.metadata == {"structured_call": True}
    assert artifact.runtime == RuntimeConfig(name="aotinductor")


def test_call_contract_maps_repeated_tensor_arguments_to_distinct_metadata():
    class SharedTensorModel(nn.Module):
        def forward(self, left, right):
            return left + right

    model = SharedTensorModel().eval()
    shared = torch.ones(1, 4)
    sample = ((shared, shared), {})
    graph_spec = _graph_spec_for(model, [sample])

    contract = PT2CallContract.capture(graph_spec, sample, model(*sample[0], **sample[1]))

    assert contract.input_order == (0, 1)


def test_unsupported_pt2_call_does_not_fail_backend_build(mocker, tmp_path):
    class ScalarArgumentModel(nn.Module):
        def forward(self, value, scale):
            return value * scale

    model = ScalarArgumentModel().eval()
    samples = [((torch.ones(1, 4), 2), {})]
    graph_spec = _graph_spec_for(model, samples)
    mocker.patch("torch.export.export", return_value=Mock())
    mocker.patch.object(torch._inductor, "aoti_compile_and_package", side_effect=_fake_aoti_compile)
    mocker.patch.object(torch._inductor, "aoti_load_package", return_value=Mock())

    backend = TorchInductorAotBackend().build(
        model,
        graph_spec,
        samples,
        device=torch.device("cpu"),
        cache_dir=tmp_path,
    )

    assert backend.is_active
    with pytest.raises(RuntimeError, match="got int"):
        backend.artifact()


def test_call_contract_capture_exception_does_not_fail_backend_build(mocker, tmp_path):
    class CallContractCaptureError(Exception):
        pass

    class CaptureFailureModel(nn.Module):
        fail = False

        def forward(self, value):
            if self.fail:
                raise CallContractCaptureError("capture failed")
            return value

    model = CaptureFailureModel().eval()
    samples = [((torch.ones(1, 4),), {})]
    graph_spec = _graph_spec_for(model, samples)
    model.fail = True
    mocker.patch("torch.export.export", return_value=Mock())
    mocker.patch.object(torch._inductor, "aoti_compile_and_package", side_effect=_fake_aoti_compile)
    mocker.patch.object(torch._inductor, "aoti_load_package", return_value=Mock())

    backend = TorchInductorAotBackend().build(
        model,
        graph_spec,
        samples,
        device=torch.device("cpu"),
        cache_dir=tmp_path,
    )

    assert backend.is_active
    with pytest.raises(RuntimeError, match="capture failed"):
        backend.artifact()


def test_build_delegates_placement_preserving_module_operations(mocker, tmp_path):
    model = ToyTorchModel().eval()
    move_module = mocker.patch("aitune.torch.backend.torch_inductor_aot_backend.move_module_to_device")
    graph_spec = model.graph_spec(batch_sizes=[1])
    sample_data = model.samples(batch_sizes=[1])
    mocker.patch("torch.export.export", return_value=Mock())
    mocker.patch.object(torch._inductor, "aoti_compile_and_package", side_effect=_fake_aoti_compile)
    mocker.patch.object(torch._inductor, "aoti_load_package", return_value=Mock())
    offload_mock = mocker.patch("aitune.torch.backend.torch_inductor_aot_backend.offload")

    TorchInductorAotBackend().build(model, graph_spec, sample_data, device=torch.device("cpu"), cache_dir=tmp_path)

    move_module.assert_called_once_with(model, torch.device("cpu"))
    offload_mock.assert_called_once_with(model, device="cpu")


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
    assert state[TorchInductorAotBackend.STATE_GRAPH_SPEC] == graph_spec.to_dict()
    assert state[TorchInductorAotBackend.STATE_PT2_CALL_CONTRACT] == {
        "input_order": (0,),
        "output_order": (0,),
        "structured": False,
    }
    assert state[TorchInductorAotBackend.STATE_SAMPLES] == sample_data.to_dict()
    assert "sample_inputs" not in state


@requires_cuda
def test_from_dict_restores_state(tmp_path, torch_device, graph_spec):
    compiled_artifact = ArtifactPath(tmp_path, "model.pt2")
    state = {
        TorchInductorAotBackend.STATE_TYPE: "TorchInductorAotBackend",
        TorchInductorAotBackend.STATE_COMPILED_MODEL_PATH: compiled_artifact,
        TorchInductorAotBackend.STATE_DEVICE: torch_device,
        TorchInductorAotBackend.STATE_GRAPH_SPEC: graph_spec.to_dict(),
        TorchInductorAotBackend.STATE_PT2_CALL_CONTRACT: {
            "input_order": (0,),
            "output_order": (0,),
            "structured": False,
        },
    }
    restored = TorchInductorAotBackend.from_dict(None, state)
    assert restored._compiled_model_artifact == compiled_artifact
    assert restored._device == torch_device
    assert restored._graph_spec == graph_spec
    assert restored._pt2_call_contract is not None
    assert restored.state == BackendState.CHECKPOINT_LOADED


@requires_cuda
def test_checkpoint_loaded_backend_reconstructs_pt2_artifact(
    mock_aoti, backend, model, graph_spec, sample_data, torch_device, tmp_path
):
    backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)
    restored = TorchInductorAotBackend.from_dict(None, backend.to_dict())

    artifact = restored.artifact()
    assert artifact.input_names == ("INPUT__0",)
    assert artifact.output_names == ("OUTPUT__0",)
    assert artifact.model.metadata == {"structured_call": False}
    assert artifact == backend.artifact()
    assert restored.state == BackendState.CHECKPOINT_LOADED

    restored.deploy(torch_device)
    assert restored.artifact() == artifact


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
