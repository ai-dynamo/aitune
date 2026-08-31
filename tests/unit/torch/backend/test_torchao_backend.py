# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from inspect import signature
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from torchao.quantization import Int8WeightOnlyConfig
from torchao.utils import is_sm_at_least_89

try:
    from torchao.quantization import PerGroup
except ImportError:
    PerGroup = None

from aitune.torch.backend.backend import BackendState, ExecutionMode
from aitune.torch.backend.torchao_backend import (
    MX_FORMATS_AVAILABLE,
    MXFP8DQ_BLOCK_SIZE_DIVISIBILITY,
    NVFP4DQ_BLOCK_SIZE_DIVISIBILITY,
    TorchAOBackend,
    TorchAOBackendConfig,
)
from aitune.torch.checkpoint.storage_tasks import torch_load_with_custom_types
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import Sample, SampleStore
from tests.toy_models.torch_models import HIDDEN_SIZE, ToyTorchModel
from tests.utilities.helpers import make_sample_store, requires_cuda

INT8DQ_OUTPUT_SIZE = 8


class TorchAOInt8DQModel(ToyTorchModel):
    """Toy model with int8dq-compatible linear dimensions."""

    def __init__(self):
        super().__init__()
        self.linear2 = nn.Linear(HIDDEN_SIZE, INT8DQ_OUTPUT_SIZE)


@pytest.fixture
def model(torch_device) -> ToyTorchModel:
    return ToyTorchModel().to(torch_device).eval()


@pytest.fixture
def sample_data(model, torch_device, tmp_path) -> SampleStore:
    return model.sample_store(tmp_path, batch_sizes=[32], device=torch_device)


def sample_data_for_model(model, torch_device) -> list[Sample]:
    sample = model.sample()
    args = (sample.to(torch_device).unsqueeze(0).repeat(32, 1),)
    kwargs = {}
    return [(args, kwargs)]


def use_quantization_test_model(quantization, model, sample_data, torch_device):
    if quantization == "int8dq":
        model = TorchAOInt8DQModel().to(torch_device).eval()
        sample_data = sample_data_for_model(model, torch_device)
    return model, sample_data


def move_to_dtype(sample_data, dtype):
    args, kwargs = sample_data[0]
    args = (args[0].to(dtype),)
    return [(args, kwargs)]


def build_backend(backend, dtype, model, sample_data, torch_device, tmp_path):
    mock_graph_spec = Mock(spec=GraphSpec)
    mock_graph_spec.name = "test"
    mock_graph_spec.input_spec = Mock()
    mock_graph_spec.input_spec.detected_dynamic_axis.return_value = False
    sample_data = move_to_dtype(sample_data, dtype)
    model.to(dtype)
    return backend.build(
        model,
        mock_graph_spec,
        make_sample_store(sample_data, tmp_path),
        device=torch_device,
        cache_dir=tmp_path,
    )


def int8_weight_only_per_group_config(group_size: int) -> Int8WeightOnlyConfig:
    if PerGroup is not None and "granularity" in signature(Int8WeightOnlyConfig).parameters:
        return Int8WeightOnlyConfig(granularity=PerGroup(group_size))
    return Int8WeightOnlyConfig(group_size=group_size)


def test_torchao_config_key():
    config = TorchAOBackendConfig(quantization="int8wo")
    key1 = config.key()
    key2 = config.key()

    assert key1 == key2


def test_torchao_supports_single_and_multi_gpu_execution():
    assert TorchAOBackend._execution_modes == frozenset({ExecutionMode.SINGLE_GPU, ExecutionMode.MULTI_GPU})


def test_torchao_config_key_includes_compile_options():
    default = TorchAOBackendConfig(quantization="int8wo")
    dynamic = TorchAOBackendConfig(quantization="int8wo", dynamic=True)
    fullgraph = TorchAOBackendConfig(quantization="int8wo", fullgraph=True)
    mode = TorchAOBackendConfig(quantization="int8wo", mode="reduce-overhead")

    assert len({default.key(), dynamic.key(), fullgraph.key(), mode.key()}) == 4


def test_torchao_config_describe():
    config = TorchAOBackendConfig(quantization="int8wo")
    describe = config.describe()
    assert describe == "quantization_config=Int8WeightOnlyConfig()"


def test_torchao_config_describe_includes_non_default_compile_options():
    config = TorchAOBackendConfig(quantization="int8wo", fullgraph=True, dynamic=False, mode="reduce-overhead")

    assert config.describe().startswith("fullgraph=True,dynamic=False,mode=reduce-overhead,")


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


def skip_if_unsupported(quantization, model=None):
    if quantization in ["fp8wo", "fp8dq"] and not is_sm_at_least_89():
        pytest.skip("fp8wo and fp8dq are not supported on this device")
    if quantization in ["mxfp8dq", "nvfp4dq"] and not MX_FORMATS_AVAILABLE:
        pytest.skip("mxfp8dq and nvfp4dq require torchao MX formats and sm100+ (Blackwell) GPU")
    if quantization == "mxfp8dq" and model is not None:
        for name, param in model.named_parameters():
            if param.ndim >= 2 and param.shape[-1] % MXFP8DQ_BLOCK_SIZE_DIVISIBILITY != 0:
                pytest.skip(
                    f"mxfp8dq requires weight last dim divisible by block_size 32: {name} has shape {param.shape}"
                )
    if quantization == "nvfp4dq" and model is not None:
        block = NVFP4DQ_BLOCK_SIZE_DIVISIBILITY
        for name, param in model.named_parameters():
            if param.ndim >= 2 and (param.shape[-1] % block != 0 or param.shape[-2] % block != 0):
                pytest.skip(f"nvfp4dq requires last 2 weight dims divisible by {block}: {name} has shape {param.shape}")


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
    model, sample_data = use_quantization_test_model(quantization, model, sample_data, torch_device)
    skip_if_unsupported(quantization, model)
    if quantization == "mxfp8dq" and dtype != torch.bfloat16:
        pytest.skip("mxfp8dq only supports bfloat16")
    if quantization == "nvfp4dq" and dtype != torch.bfloat16:
        pytest.skip("nvfp4dq only supports bfloat16")

    config = TorchAOBackendConfig(quantization=quantization)
    backend = TorchAOBackend(config=config)
    do_test_backend(backend, dtype, model, sample_data, torch_device, tmp_path)


@requires_cuda
@pytest.mark.parametrize(
    "quantization_config",
    [int8_weight_only_per_group_config(group_size=16)],
    ids=["int8wo with different group size"],
)
def test_torchao_backend_build_with_user_config(quantization_config, model, sample_data, torch_device, tmp_path):
    config = TorchAOBackendConfig(quantization_config=quantization_config)
    backend = TorchAOBackend(config=config)
    do_test_backend(backend, torch.bfloat16, model, sample_data, torch_device, tmp_path)


def test_invalid_quantization_type():
    with pytest.raises(ValueError):
        TorchAOBackendConfig(quantization="invalid_type")  # type: ignore


def test_torchao_config_rejects_invalid_mode():
    with pytest.raises(ValueError, match="Invalid mode"):
        TorchAOBackendConfig(quantization="int8wo", mode="invalid")  # pytype: disable=wrong-arg-types


def test_build_auto_dynamic_does_not_mutate_config(mocker, tmp_path):
    model = ToyTorchModel().eval()
    graph_spec = model.graph_spec(batch_sizes=[1, 2])
    sample_data = model.sample_store(tmp_path, batch_sizes=[1])
    config = TorchAOBackendConfig(quantization="int8wo", dynamic=None)
    backend = TorchAOBackend(config=config)
    original_key = backend.key()
    compile_mock = mocker.patch("aitune.torch.backend.torchao_backend.torch.compile", return_value=model)
    mocker.patch("aitune.torch.backend.torchao_backend.quantize_")
    move_module_mock = mocker.patch("aitune.torch.backend.torchao_backend.move_module_to_device")

    backend.build(
        model,
        graph_spec,
        sample_data,
        device=torch.device("cpu"),
        cache_dir=tmp_path,
    )

    assert config.dynamic is None
    assert backend.key() == original_key
    assert compile_mock.call_args.kwargs["dynamic"] is True
    move_module_mock.assert_called_once_with(model, "cpu")


def test_auto_dynamic_setting_is_restored_from_state_dict(mocker, tmp_path):
    model = ToyTorchModel().eval()
    graph_spec = model.graph_spec(batch_sizes=[1, 2])
    sample_data = model.sample_store(tmp_path, batch_sizes=[1])
    config = TorchAOBackendConfig(quantization="int8wo", dynamic=None)
    backend = TorchAOBackend(config=config)
    compile_mock = mocker.patch("aitune.torch.backend.torchao_backend.torch.compile", return_value=model)
    mocker.patch("aitune.torch.backend.torchao_backend.quantize_")

    backend.build(
        model,
        graph_spec,
        sample_data,
        device=torch.device("cpu"),
        cache_dir=tmp_path,
    )
    state_dict = backend.to_dict()
    loaded_backend = TorchAOBackend.from_dict(model, state_dict)

    assert state_dict[TorchAOBackend.STATE_CONFIG]["dynamic"] is None
    assert state_dict[TorchAOBackend.STATE_COMPILE_DYNAMIC] is True

    compile_mock.reset_mock()
    loaded_backend.activate()

    assert compile_mock.call_args.kwargs["dynamic"] is True


def test_activate_runs_compatibility_preflight(mocker):
    model = ToyTorchModel().eval()
    config = TorchAOBackendConfig(quantization="nvfp4dq")
    backend = TorchAOBackend(config=config)
    backend._orig_module = model
    backend._data = []
    backend._device = torch.device("cpu")
    backend.state = BackendState.CHECKPOINT_LOADED
    quantize_mock = mocker.patch("aitune.torch.backend.torchao_backend.quantize_")
    mocker.patch("aitune.torch.backend.torchao_backend.MX_FORMATS_AVAILABLE", False)

    with pytest.raises(RuntimeError):
        backend.activate()

    quantize_mock.assert_not_called()


def test_filter_fn_is_honored_by_nvfp4dq_compatibility_check():
    if not MX_FORMATS_AVAILABLE:
        pytest.skip("nvfp4dq compatibility checks require torchao MX formats and sm100+ (Blackwell) GPU")

    model = ToyTorchModel().eval().to(torch.bfloat16)

    unfiltered_backend = TorchAOBackend(config=TorchAOBackendConfig(quantization="nvfp4dq"))
    with pytest.raises(RuntimeError, match=r"linear2\.weight"):
        unfiltered_backend._check_hardware_compatibility(model)

    filtered_config = TorchAOBackendConfig(
        quantization="nvfp4dq",
        filter_fn=lambda _module, name: name == "linear1",
    )
    backend = TorchAOBackend(config=filtered_config)

    backend._check_hardware_compatibility(model)


@requires_cuda
@pytest.mark.parametrize("quantization", TorchAOBackendConfig._QUANTIZATION_CONFIGS.keys())
def test_serialization(quantization, tmp_path, model, sample_data, torch_device):
    model, sample_data = use_quantization_test_model(quantization, model, sample_data, torch_device)
    skip_if_unsupported(quantization, model)
    dtype = torch.bfloat16 if quantization in ("mxfp8dq", "nvfp4dq") else torch.float16

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
    original = TorchAOBackendConfig(quantization="int8wo", fullgraph=True, dynamic=True, mode="reduce-overhead")
    restored = TorchAOBackendConfig.from_dict(original.to_dict())
    assert isinstance(restored.quantization_config, type(original.quantization_config))
    assert restored.fullgraph is True
    assert restored.dynamic is True
    assert restored.mode == "reduce-overhead"
