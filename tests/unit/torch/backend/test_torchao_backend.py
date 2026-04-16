# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from torchao.quantization import Int8WeightOnlyConfig
from torchao.utils import is_sm_at_least_89

from aitune.torch.backend.torchao_backend import TorchAOBackend, TorchAOBackendConfig
from aitune.torch.checkpoint.storage_tasks import torch_load_with_custom_types
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from tests.toy_models.torch_models import ToyTorchModel
from tests.utilities.helpers import requires_cuda


@pytest.fixture
def model(torch_device) -> nn.Module:
    return ToyTorchModel().to(torch_device).eval()


@pytest.fixture
def sample_data(model, torch_device) -> list[Sample]:
    sample = model.sample()
    args = (sample.to(torch_device).unsqueeze(0).repeat(32, 1),)
    kwargs = {}
    return [(args, kwargs)]


def move_to_dtype(sample_data, dtype):
    args, kwargs = sample_data[0]
    args = (args[0].to(dtype),)
    return [(args, kwargs)]


def build_backend(backend, dtype, model, sample_data, torch_device, tmp_path):
    mock_graph_spec = Mock(spec=GraphSpec)
    mock_graph_spec.name = "test"
    sample_data = move_to_dtype(sample_data, dtype)
    model.to(dtype)
    return backend.build(model, mock_graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)


def test_torchao_config_key():
    config = TorchAOBackendConfig(quantization="int8wo")
    key1 = config.key()
    key2 = config.key()

    assert key1 == key2


def test_torchao_config_describe():
    config = TorchAOBackendConfig(quantization="int8wo")
    describe = config.describe()
    assert describe == "quantization_config=Int8WeightOnlyConfig()"


def test_torchao_config_initialization():
    # Test valid initialization with quantization type
    for quantization in TorchAOBackendConfig._QUANTIZATION_CONFIGS.keys():
        TorchAOBackendConfig(quantization=quantization)  # type: ignore

    # Test valid initialization with quantization config
    config = Int8WeightOnlyConfig()
    TorchAOBackendConfig(quantization_config=config)

    # Test invalid initialization with both parameters
    with pytest.raises(ValueError, match="Only one of quantization or quantization_config should be provided."):
        TorchAOBackendConfig(quantization="int8wo", quantization_config=config)

    # Test invalid initialization with neither parameter
    with pytest.raises(ValueError, match="Either quantization or quantization_config should be provided."):
        TorchAOBackendConfig()


def do_test_backend(backend, dtype, model, sample_data, torch_device, tmp_path):
    """Helper function to test backend with given dtype data.

    Args:
        backend: The backend instance to test
        dtype: The dtype to use for the test
    """
    backend = build_backend(backend, dtype, model, sample_data, torch_device, tmp_path)
    sample_data = move_to_dtype(sample_data, dtype)
    args, kwargs = sample_data[0]

    # then
    assert backend is not None
    try:
        backend.infer(*args, **kwargs)
    finally:
        backend.deactivate()


@requires_cuda
@pytest.mark.parametrize("quantization", TorchAOBackendConfig._QUANTIZATION_CONFIGS.keys())
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float16, torch.float32],
    ids=["bfloat16", "float16", "float32"],
)
def test_torchao_backend_build(quantization, dtype, model, sample_data, torch_device, tmp_path):
    if quantization in ["fp8wo", "fp8dq"] and not is_sm_at_least_89():
        pytest.skip("fp8wo and fp8dq are not supported on this device")

    config = TorchAOBackendConfig(quantization=quantization)
    backend = TorchAOBackend(config=config)
    do_test_backend(backend, dtype, model, sample_data, torch_device, tmp_path)


@requires_cuda
@pytest.mark.parametrize(
    "quantization_config",
    [Int8WeightOnlyConfig(group_size=16)],
    ids=["int8wo with different group size"],
)
def test_torchao_backend_build_with_user_config(quantization_config, model, sample_data, torch_device, tmp_path):
    config = TorchAOBackendConfig(quantization_config=quantization_config)
    backend = TorchAOBackend(config=config)
    do_test_backend(backend, torch.bfloat16, model, sample_data, torch_device, tmp_path)


def test_invalid_quantization_type():
    with pytest.raises(ValueError):
        TorchAOBackendConfig(quantization="invalid_type")  # type: ignore


@requires_cuda
@pytest.mark.parametrize("quantization", TorchAOBackendConfig._QUANTIZATION_CONFIGS.keys())
def test_serialization(quantization, tmp_path, model, sample_data, torch_device):
    dtype = torch.float16
    if quantization in ["fp8wo", "fp8dq"] and not is_sm_at_least_89():
        pytest.skip("fp8wo and fp8dq are not supported on this device")

    if quantization == "fp8dq" and torch.cuda.get_device_capability() == (9, 0):
        pytest.skip("fp8dq is unstable on H100")

    config = TorchAOBackendConfig(quantization=quantization)
    backend = build_backend(TorchAOBackend(config=config), dtype, model, sample_data, torch_device, tmp_path)
    sample_data = move_to_dtype(sample_data, dtype)
    args, kwargs = sample_data[0]
    loaded_backend = None
    try:
        expected = backend.infer(*args, **kwargs)
        model.to("cuda")

        state_dict = backend.to_dict()  # type: ignore

        torch.save(state_dict, tmp_path / "state_dict.pth")
        backend.deactivate()
        backend = None

        loaded_backend = TorchAOBackend.from_dict(model, torch_load_with_custom_types(tmp_path / "state_dict.pth"))
        loaded_backend.activate()
        actual = loaded_backend.infer(*args, **kwargs)
        torch.testing.assert_close(expected, actual)
        loaded_backend.deactivate()
        loaded_backend = None
    except Exception as e:
        # do cleanup in case of an exception, torchao backend is susceptible to memory issues when not deactivated
        if backend:
            backend.deactivate()
        if loaded_backend:
            loaded_backend.deactivate()
        raise e


# --- TorchAOBackendConfig.from_dict ---


def test_torchao_config_from_dict_yaml_path_string_quantization():
    config = TorchAOBackendConfig.from_dict({"quantization": "fp8wo"})
    assert config.quantization == "fp8wo"
    assert config.quantization_config is not None


def test_torchao_config_from_dict_yaml_path_int8wo():
    config = TorchAOBackendConfig.from_dict({"quantization": "int8wo"})
    assert config.quantization == "int8wo"


def test_torchao_config_from_dict_checkpoint_path_bytes():
    from dill import dumps
    from torchao.quantization import Float8WeightOnlyConfig

    quant_config = Float8WeightOnlyConfig()
    data = {"quantization_config": dumps(quant_config)}
    config = TorchAOBackendConfig.from_dict(data)
    assert config.quantization_config is not None


def test_torchao_config_from_dict_round_trip():
    original = TorchAOBackendConfig(quantization="int8wo")
    restored = TorchAOBackendConfig.from_dict(original.to_dict())
    assert isinstance(restored.quantization_config, type(original.quantization_config))
