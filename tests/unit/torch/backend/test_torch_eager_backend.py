# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TorchInductorJitBackend."""

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from aitune.torch.backend.torch_eager import TorchEagerBackend, TorchEagerBackendConfig
from aitune.torch.checkpoint.storage_tasks import TorchLoadTask, TorchSaveTask
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import Sample
from tests.toy_models import ToyTorchModel
from tests.utilities.helpers import requires_cuda

IN_FEATURES = 32
OUT_FEATURES = 5
BATCH_SIZE = 2


@pytest.fixture
def model(torch_device) -> nn.Module:
    return ToyTorchModel().to(torch_device).eval()


@pytest.fixture
def sample_data(model, torch_device) -> list[Sample]:
    sample = model.sample()
    args = (sample.to(torch_device).unsqueeze(0),)
    kwargs = {}
    return [(args, kwargs)]


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
    backend = backend.build(model, graph_spec=mock_graph_spec, samples=data, device=device, cache_dir=tmp_path)
    return backend


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

    # Test inference
    output = backend.infer(*args, **kwargs)
    assert output.shape[0] == 1
    assert output.dtype == dtype

    # Test deactivation
    backend.deactivate()

    # Test serialization and deployment
    state_dict = backend.to_dict()
    loaded_backend = TorchEagerBackend.from_dict(model, state_dict)
    loaded_backend.deploy("cuda")
    loaded_output = loaded_backend.infer(*args, **kwargs)
    assert output.dtype == loaded_output.dtype


@requires_cuda
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
    ids=["float16", "bfloat16", "float32"],
)
def test_torch_eager_backend_build(dtype, model, sample_data, tmp_path):
    """Test backend build with different dtypes."""
    config = TorchEagerBackendConfig()
    backend = TorchEagerBackend(config=config)
    do_test_backend(backend, dtype, model, sample_data, tmp_path)


@requires_cuda
@pytest.mark.parametrize(
    "autocast_dtype",
    [torch.float16, torch.bfloat16],
    ids=["float16", "bfloat16"],
)
def test_torch_eager_backend_with_autocast(autocast_dtype, model, sample_data, tmp_path):
    """Test backend with autocast enabled."""
    config = TorchEagerBackendConfig(autocast_enabled=True, autocast_dtype=autocast_dtype)
    backend = TorchEagerBackend(config=config)
    do_test_backend(backend, torch.float32, model, sample_data, tmp_path)


@requires_cuda
def test_serialization(model, sample_data, tmp_path):
    backend = backend_build(TorchEagerBackend(), torch.float16, model, sample_data, tmp_path)
    state_dict = backend.to_dict()  # type: ignore

    TorchSaveTask().save(tmp_path, state_dict)
    state_dict = TorchLoadTask().load(tmp_path)
    loaded_backend = TorchEagerBackend.from_dict(model, state_dict)

    loaded_backend.activate()
    sample_data = move_to_dtype(sample_data, torch.float16)
    args, kwargs = sample_data[0]
    torch.testing.assert_close(backend.infer(*args, **kwargs), loaded_backend.infer(*args, **kwargs))


# --- TorchEagerBackendConfig.from_dict ---


def test_eager_config_from_dict_defaults():
    config = TorchEagerBackendConfig.from_dict({})
    assert config == TorchEagerBackendConfig()


def test_eager_config_from_dict_custom_fields():
    config = TorchEagerBackendConfig.from_dict({"autocast_enabled": True, "autocast_dtype": torch.float16})
    assert config.autocast_enabled is True
    assert config.autocast_dtype == torch.float16


def test_eager_config_from_dict_round_trip():
    original = TorchEagerBackendConfig(autocast_enabled=True)
    restored = TorchEagerBackendConfig.from_dict(original.to_dict())
    assert restored == original
