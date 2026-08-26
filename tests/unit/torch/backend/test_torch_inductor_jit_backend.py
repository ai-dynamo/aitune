# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TorchInductorJitBackend."""

from unittest.mock import Mock

import pytest
import torch

from aitune.torch.backend.torch_inductor_jit_backend import TorchInductorJitBackend, TorchInductorJitBackendConfig
from aitune.torch.checkpoint.storage_tasks import TorchLoadTask, TorchSaveTask
from aitune.torch.libs.torch_compile import TorchCompileMode
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import SampleStore
from tests.toy_models import ToyTorchModel
from tests.utilities.helpers import make_sample_store, requires_cuda

IN_FEATURES = 32
OUT_FEATURES = 5
BATCH_SIZE = 2


@pytest.fixture
def model(torch_device) -> ToyTorchModel:
    return ToyTorchModel().to(torch_device).eval()


@pytest.fixture
def sample_data(model, torch_device, tmp_path) -> SampleStore:
    return model.sample_store(tmp_path, batch_sizes=[1], device=torch_device)


def move_to_dtype(sample_data, dtype):
    args, kwargs = sample_data[0]
    args = (args[0].to(dtype),)
    return [(args, kwargs)]


def backend_build(backend, dtype, model, sample_data, tmp_path, device="cuda"):
    """Build the model with the backend."""
    device = torch.device(device)
    model = model.to(device, dtype=dtype)
    data = move_to_dtype(sample_data, dtype)
    mock_graph_spec = Mock(spec=GraphSpec)
    mock_graph_spec.name = "test"
    mock_graph_spec.input_spec = Mock()
    mock_graph_spec.input_spec.detected_dynamic_axis.return_value = False
    backend = backend.build(
        model, graph_spec=mock_graph_spec, samples=make_sample_store(data, tmp_path), device=device, cache_dir=tmp_path
    )
    return backend


def test_torch_inductor_build_auto_dynamic_does_not_mutate_config(mocker, tmp_path):
    toy = ToyTorchModel().eval()
    graph_spec = toy.graph_spec(batch_sizes=[1, 2])
    sample_data = toy.sample_store(tmp_path, batch_sizes=[1])
    compile_mock = mocker.patch("aitune.torch.backend.torch_inductor_jit_backend.torch.compile", return_value=toy)
    backend = TorchInductorJitBackend(TorchInductorJitBackendConfig(dynamic=None))
    original_key = backend.key()

    backend.build(
        toy,
        graph_spec=graph_spec,
        samples=sample_data,
        device=torch.device("cpu"),
        cache_dir=tmp_path,
    )

    assert backend._config.dynamic is None
    assert backend.key() == original_key
    assert compile_mock.call_args.kwargs["dynamic"] is True


def test_torch_inductor_build_keeps_sample_store_without_retaining_data(mocker, tmp_path):
    toy = ToyTorchModel().eval()
    graph_spec = toy.graph_spec(batch_sizes=[1])
    sample_data = toy.sample_store(tmp_path, batch_sizes=[1])
    mocker.patch("aitune.torch.backend.torch_inductor_jit_backend.torch.compile", return_value=toy)

    backend = TorchInductorJitBackend().build(
        toy,
        graph_spec=graph_spec,
        samples=sample_data,
        device=torch.device("cpu"),
        cache_dir=tmp_path,
    )

    assert isinstance(backend._samples, SampleStore)
    assert backend._data is None
    state = backend.to_dict()
    assert state[TorchInductorJitBackend.STATE_SAMPLES] == backend._samples.to_dict()
    assert TorchInductorJitBackend.STATE_DATA not in state


def test_torch_inductor_loads_legacy_inline_samples(mocker, tmp_path):
    toy = ToyTorchModel().eval()
    graph_spec = toy.graph_spec(batch_sizes=[1])
    inline_samples = toy.samples(batch_sizes=[1])
    samples = make_sample_store(inline_samples, tmp_path)
    compile_mock = mocker.patch("aitune.torch.backend.torch_inductor_jit_backend.torch.compile", return_value=toy)
    backend = TorchInductorJitBackend()
    backend.build(
        toy,
        graph_spec=graph_spec,
        samples=samples,
        device=torch.device("cpu"),
        cache_dir=tmp_path,
    )
    state = backend.to_dict()
    state.pop(TorchInductorJitBackend.STATE_SAMPLES)
    state[TorchInductorJitBackend.STATE_DATA] = inline_samples

    loaded_backend = TorchInductorJitBackend.from_dict(toy, state)
    compile_mock.reset_mock()
    loaded_backend.activate()

    compile_mock.assert_called_once()


def test_torch_inductor_auto_dynamic_setting_is_restored_from_state_dict(mocker, tmp_path):
    toy = ToyTorchModel().eval()
    graph_spec = toy.graph_spec(batch_sizes=[1, 2])
    sample_data = toy.sample_store(tmp_path, batch_sizes=[1])
    compile_mock = mocker.patch("aitune.torch.backend.torch_inductor_jit_backend.torch.compile", return_value=toy)
    backend = TorchInductorJitBackend(TorchInductorJitBackendConfig(dynamic=None))

    backend.build(
        toy,
        graph_spec=graph_spec,
        samples=sample_data,
        device=torch.device("cpu"),
        cache_dir=tmp_path,
    )
    state_dict = backend.to_dict()
    loaded_backend = TorchInductorJitBackend.from_dict(toy, state_dict)

    assert state_dict[TorchInductorJitBackend.STATE_CONFIG]["dynamic"] is None
    assert state_dict[TorchInductorJitBackend.STATE_COMPILE_DYNAMIC] is True

    compile_mock.reset_mock()
    loaded_backend.activate()

    assert compile_mock.call_args.kwargs["dynamic"] is True


def test_torch_inductor_build_preserves_explicit_dynamic_false(mocker, tmp_path):
    toy = ToyTorchModel().eval()
    graph_spec = toy.graph_spec(batch_sizes=[1, 2])
    sample_data = toy.sample_store(tmp_path, batch_sizes=[1])
    compile_mock = mocker.patch("aitune.torch.backend.torch_inductor_jit_backend.torch.compile", return_value=toy)
    backend = TorchInductorJitBackend(TorchInductorJitBackendConfig(dynamic=False))

    backend.build(
        toy,
        graph_spec=graph_spec,
        samples=sample_data,
        device=torch.device("cpu"),
        cache_dir=tmp_path,
    )

    assert backend._config.dynamic is False
    assert compile_mock.call_args.kwargs["dynamic"] is False


def do_test_backend(backend, dtype, model, sample_data, tmp_path):
    """Helper function to test backend with given dtype and device.

    Args:
        backend: The backend instance to test
        dtype: The dtype to use for the test
        device: The device to use for the test
    """
    backend = backend_build(backend, dtype, model, sample_data, tmp_path)

    sample_data = move_to_dtype(sample_data, dtype)
    args, kwargs = sample_data[0]

    # Verify the backend is properly initialized
    assert backend is not None
    assert hasattr(backend, "_compiled_module")

    # Test inference
    output = backend.infer(*args, **kwargs)
    assert output.shape[0] == 1
    assert output.dtype == dtype

    # Test deactivation
    backend.deactivate()

    # Test serialization and deployment
    state_dict = backend.to_dict()
    loaded_backend = TorchInductorJitBackend.from_dict(model, state_dict)
    loaded_backend.deploy(torch.device("cuda"))
    loaded_output = loaded_backend.infer(*args, **kwargs)
    assert output.dtype == loaded_output.dtype


@requires_cuda
@pytest.mark.parametrize(
    "mode",
    ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
    ids=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
    ids=["float16", "bfloat16", "float32"],
)
def test_torch_inductor_backend_build(mode: TorchCompileMode, dtype, model, sample_data, tmp_path):
    """Test backend build with different modes and dtypes."""
    config = TorchInductorJitBackendConfig(mode=mode)
    backend = TorchInductorJitBackend(config=config)
    do_test_backend(backend, dtype, model, sample_data, tmp_path)


@requires_cuda
def test_torch_inductor_backend_with_options(model, sample_data, tmp_path):
    """Test backend with custom options."""
    # Testing with some common inductor options
    options = {
        "max_autotune": True,
        "aggressive_fusion": True,
        "debug": True,
    }
    config = TorchInductorJitBackendConfig(options=options)
    backend = TorchInductorJitBackend(config=config)
    do_test_backend(backend, torch.float32, model, sample_data, tmp_path)


@requires_cuda
@pytest.mark.parametrize(
    "autocast_dtype",
    [torch.float16, torch.bfloat16],
    ids=["float16", "bfloat16"],
)
def test_torch_inductor_backend_with_autocast(autocast_dtype, model, sample_data, tmp_path):
    """Test backend with autocast enabled."""
    config = TorchInductorJitBackendConfig(autocast_enabled=True, autocast_dtype=autocast_dtype)
    backend = TorchInductorJitBackend(config=config)
    do_test_backend(backend, torch.float32, model, sample_data, tmp_path)


def test_torch_inductor_backend_with_mode_and_options():
    """Test backend with mode and options."""
    with pytest.raises(ValueError, match="Cannot specify both 'mode' and 'options' parameters in config. "):
        config = TorchInductorJitBackendConfig(mode="max-autotune", options={"max_autotune": True})
        TorchInductorJitBackend(config=config)


def test_torch_inductor_config_rejects_invalid_mode():
    with pytest.raises(ValueError, match="Invalid mode"):
        TorchInductorJitBackendConfig(mode="invalid")  # pytype: disable=wrong-arg-types


@requires_cuda
def test_serialization(model, sample_data, tmp_path):
    backend = backend_build(TorchInductorJitBackend(), torch.float16, model, sample_data, tmp_path)
    state_dict = backend.to_dict()  # type: ignore

    TorchSaveTask().save(tmp_path, state_dict)
    state_dict = TorchLoadTask().load(tmp_path)
    loaded_backend = TorchInductorJitBackend.from_dict(model, state_dict)

    loaded_backend.activate()
    sample_data = move_to_dtype(sample_data, torch.float16)
    args, kwargs = sample_data[0]
    torch.testing.assert_close(backend.infer(*args, **kwargs), loaded_backend.infer(*args, **kwargs))


# --- TorchInductorJitBackendConfig.from_dict ---


def test_inductor_jit_config_from_dict_defaults():
    config = TorchInductorJitBackendConfig.from_dict({})
    assert config == TorchInductorJitBackendConfig()


def test_inductor_jit_config_from_dict_custom_fields():
    config = TorchInductorJitBackendConfig.from_dict({"fullgraph": True, "mode": "max-autotune"})
    assert config.fullgraph is True
    assert config.mode == "max-autotune"


def test_inductor_jit_config_from_dict_round_trip():
    original = TorchInductorJitBackendConfig(fullgraph=True, dynamic=True)
    restored = TorchInductorJitBackendConfig.from_dict(original.to_dict())
    assert restored == original
