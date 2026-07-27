# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
import torch
import torch.nn as nn

from aitune.torch.backend.torch_tensorrt_aot_backend import (
    TorchTensorRTAotBackend,
    TorchTensorRTAotBackendConfig,
)
from aitune.torch.checkpoint.storage_tasks import torch_load_with_custom_types
from aitune.torch.dynamic_shapes import BatchDim
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.utils.tensor import format_tensor_name
from tests.toy_models import ToyTorchModel
from tests.toy_models.torch_models import ToyTorchConditionalModel
from tests.utilities.helpers import make_graph_spec, requires_cuda, update_input_spec


@dataclass
class TorchTensorRTTestConfig:
    workspace_size: int = 0


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def module(self):
        return self


@pytest.fixture
def model(torch_device) -> SimpleModel:
    return SimpleModel().to(torch_device).eval()


@pytest.fixture
def sample_data(torch_device) -> list[Sample]:
    return [
        ((torch.randn(1, 10, device=torch_device),), {}),
        ((torch.randn(4, 10, device=torch_device),), {}),
        ((torch.randn(2, 10, device=torch_device),), {}),
    ]


@pytest.fixture
def graph_spec(model, sample_data) -> GraphSpec:
    args, kwargs = sample_data[0]
    output = model(*args, **kwargs)
    return make_graph_spec(model.forward, (args, kwargs), output, name="0", strict=True)


@pytest.fixture
def backend(mocker) -> TorchTensorRTAotBackend:
    from aitune.torch.backend.torch_tensorrt_aot_backend import torch_tensorrt

    if torch_tensorrt is None:
        pytest.skip("torch_tensorrt is not available")

    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.assert_cuda_is_available")  # always available

    return TorchTensorRTAotBackend(config=TorchTensorRTAotBackendConfig(compile_config=TorchTensorRTTestConfig()))


def _fake_torch_tensorrt_save(model, path, **kwargs):
    Path(path).write_bytes(b"fake")


@pytest.fixture
def mock_torch_tensorrt(mocker, model: SimpleModel):
    mock_torch_tensorrt = mocker.Mock()
    mock_torch_tensorrt.dynamo.compile = mocker.Mock(return_value=model)
    mock_torch_tensorrt.save = mocker.Mock(side_effect=_fake_torch_tensorrt_save)

    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.torch_tensorrt", mock_torch_tensorrt)
    return mock_torch_tensorrt


def _graph_spec_from_samples(model: nn.Module, samples: list[Sample]) -> GraphSpec:
    args, kwargs = samples[0]
    output = model(*args, **kwargs)
    graph_spec = make_graph_spec(model.forward, (args, kwargs), output, name="0", strict=True)
    for args, kwargs in samples[1:]:
        output = model(*args, **kwargs)
        update_input_spec(graph_spec, (args, kwargs), strict=True)
        graph_spec.output_spec.update_shapes_seen(SampleMetadata.from_outputs(output))
    return graph_spec


@requires_cuda
def test_torch_tensorrt_aot_backend_config_key():
    """Test backend config with cache_dir."""

    config = TorchTensorRTAotBackendConfig()
    key1 = config.key()
    key2 = config.key()

    assert key1 == key2


@requires_cuda
def test_torch_tensorrt_aot_backend_config_describe():
    """Test backend config with cache_dir."""
    config = TorchTensorRTAotBackendConfig(compile_config=TorchTensorRTTestConfig())
    describe = config.describe()

    assert describe == "compile_config=TorchTensorRTConfig()"

    config = TorchTensorRTAotBackendConfig(compile_config=TorchTensorRTTestConfig(workspace_size=1))
    describe = config.describe()

    assert describe == "compile_config=TorchTensorRTConfig(workspace_size=1)"

    config = TorchTensorRTAotBackendConfig(compile_config=TorchTensorRTTestConfig(), pickle_protocol=1)
    describe = config.describe()

    assert describe == "compile_config=TorchTensorRTConfig(),pickle_protocol=1"


@requires_cuda
def test_mock_build(
    mocker,
    mock_torch_tensorrt,
    backend: TorchTensorRTAotBackend,
    model: SimpleModel,
    graph_spec: GraphSpec,
    sample_data: list[Sample],
    torch_device: torch.device,
    tmp_path: Path,
):
    sentinel_exported = mocker.Mock(name="ExportedProgram")
    export_mock = mocker.patch("torch.export.export", return_value=sentinel_exported)

    active_backend = backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)

    assert active_backend is backend

    active_backend = cast(TorchTensorRTAotBackend, active_backend)
    assert active_backend._device == torch_device
    assert active_backend.is_active

    export_mock.assert_called_once()
    locator, tensor_spec = graph_spec.input_spec.tensor_data[0]
    mock_torch_tensorrt.Input.assert_called_once_with(
        min_shape=tensor_spec.min_shape,
        opt_shape=tensor_spec.max_shape,
        max_shape=tensor_spec.max_shape,
        dtype=tensor_spec.dtype,
        name=format_tensor_name(locator.path, "input"),
    )
    mock_torch_tensorrt.dynamo.compile.assert_called_once()
    # Pin the pipeline shape: the ExportedProgram from torch.export.export must be
    # forwarded as the first positional arg to torch_tensorrt.dynamo.compile.
    assert mock_torch_tensorrt.dynamo.compile.call_args[0][0] is sentinel_exported


@pytest.mark.parametrize(
    ("dynamic_shapes", "expected_opt", "expected_max"),
    [
        pytest.param(None, 4, 4, id="inferred"),
        pytest.param(
            {"x": (BatchDim("batch", min=1, opt=2, max=8), 10)},
            2,
            8,
            id="explicit",
        ),
    ],
)
def test_mock_build_exports_bounded_dynamic_shapes(
    mocker,
    tmp_path: Path,
    dynamic_shapes,
    expected_opt,
    expected_max,
):
    model = SimpleModel().eval()
    sample_data = [
        ((torch.randn(1, 10),), {}),
        ((torch.randn(4, 10),), {}),
        ((torch.randn(2, 10),), {}),
    ]
    graph_spec = _graph_spec_from_samples(model, sample_data)
    if dynamic_shapes is not None:
        graph_spec.dynamic_shapes = dynamic_shapes
    mock_torch_tensorrt = mocker.Mock()
    mock_torch_tensorrt.dynamo.compile = mocker.Mock(return_value=model)
    mock_torch_tensorrt.save = mocker.Mock(side_effect=_fake_torch_tensorrt_save)
    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.torch_tensorrt", mock_torch_tensorrt)
    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.assert_cuda_is_available")
    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.get_cuda_device", return_value=0)
    mocker.patch.object(TorchTensorRTAotBackend, "_devices", ["cpu", "cuda"])

    backend = TorchTensorRTAotBackend(config=TorchTensorRTAotBackendConfig(compile_config=TorchTensorRTTestConfig()))
    sentinel_exported = mocker.Mock(name="ExportedProgram")
    export_mock = mocker.patch("torch.export.export", return_value=sentinel_exported)

    backend.build(model, graph_spec, sample_data, device=torch.device("cpu"), cache_dir=tmp_path)

    dynamic_shapes = export_mock.call_args.kwargs["dynamic_shapes"]
    dim = dynamic_shapes["x"][0]
    assert dim is not torch.export.Dim.AUTO
    assert dim.min == 1
    assert dim.max == expected_max
    mock_torch_tensorrt.Input.assert_called_once_with(
        min_shape=[1, 10],
        opt_shape=[expected_opt, 10],
        max_shape=[expected_max, 10],
        dtype=torch.float32,
        name=format_tensor_name("x", "input"),
    )


@requires_cuda
def test_mock_infer(
    mocker,
    model: SimpleModel,
    graph_spec: GraphSpec,
    sample_data: list[Sample],
    backend: TorchTensorRTAotBackend,
    torch_device: torch.device,
    tmp_path: Path,
):
    mocker.patch("torch.export.export")
    mocker.patch("torch.export.save")
    load = mocker.patch("torch.export.load", return_value=model)
    mocker.patch(
        "aitune.torch.backend.torch_tensorrt_aot_backend.torch_tensorrt.save",
        side_effect=_fake_torch_tensorrt_save,
    )
    # torch.export.export is mocked above (returns Mock); torch_tensorrt.dynamo.compile
    # would reject a Mock at its isinstance(_, ExportedProgram) check, so mock it too
    # and return the original model — _opt_module then bypasses real TRT compilation
    # while infer() still produces a real (1, 2) output via the underlying nn.Linear.
    mocker.patch(
        "aitune.torch.backend.torch_tensorrt_aot_backend.torch_tensorrt.dynamo.compile",
        return_value=model,
    )

    backend = backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)

    args, kwargs = sample_data[0]

    backend = cast(TorchTensorRTAotBackend, backend)

    output = backend.infer(*args, **kwargs)
    assert output.shape == (1, 2)

    load.assert_not_called()


@requires_cuda
def test_build_with_custom_precisions(
    model: SimpleModel,
    graph_spec: GraphSpec,
    sample_data: list[Sample],
    mocker,
    mock_torch_tensorrt,
    torch_device: torch.device,
    tmp_path: Path,
):
    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.assert_cuda_is_available")

    config = TorchTensorRTAotBackendConfig(compile_config=TorchTensorRTTestConfig(workspace_size=1))
    backend = TorchTensorRTAotBackend(
        config=config,
    )

    backend = backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)
    assert backend is not None

    mock_torch_tensorrt.dynamo.compile.assert_called_once()
    assert mock_torch_tensorrt.dynamo.compile.call_args[1]["workspace_size"] == 1


@requires_cuda
def test_full_run_simple_model(
    backend: TorchTensorRTAotBackend,
    model: SimpleModel,
    graph_spec: GraphSpec,
    sample_data: list[Sample],
    torch_device: torch.device,
    tmp_path: Path,
):
    backend = backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)

    assert backend._exported_model_path.exists()

    args, kwargs = sample_data[0]

    backend = cast(TorchTensorRTAotBackend, backend)
    assert backend._opt_module is not None

    output = backend.infer(*args, **kwargs)
    assert output.shape == (1, 2)


@requires_cuda
def test_full_run_toy_model_linear(torch_device: torch.device, tmp_path: Path):
    model = ToyTorchModel().to(torch_device)
    samples = model.samples(batch_sizes=[2], device=torch_device)
    graph_spec = model.graph_spec(batch_sizes=[2], device=torch_device)

    backend = TorchTensorRTAotBackend()
    backend = backend.build(model, graph_spec, samples, device=torch_device, cache_dir=tmp_path)
    args, kwargs = samples[0]

    backend = cast(TorchTensorRTAotBackend, backend)
    assert backend._opt_module is not None

    output = backend.infer(*args, **kwargs)
    assert output.shape == (2, 5)


@pytest.mark.skip(reason="Currently not supported by Torch-TRT https://github.com/pytorch/TensorRT/issues/767")
@requires_cuda
def test_conditional_model(torch_device, tmp_path: Path):
    model = ToyTorchConditionalModel().to(torch_device)
    inputs = model.inputs(batch_sizes=[2], device=torch_device)
    samples = model.samples(batch_sizes=[2], device=torch_device, kwargs={"apply_relu": True})
    graph_spec = model.graph_spec(batch_sizes=[2], device=torch_device, kwargs={"apply_relu": True})

    backend = TorchTensorRTAotBackend()
    backend = backend.build(model, graph_spec, samples, device=torch_device, cache_dir=tmp_path)

    assert backend._opt_module is not None

    output = backend.infer(inputs, apply_relu=True)
    assert output.shape == (2, 5)


@requires_cuda
def test_serialization(torch_device, tmp_path):
    model = ToyTorchModel().to(torch_device)
    samples = model.samples(batch_sizes=[2], device=torch_device)
    graph_spec = model.graph_spec(batch_sizes=[2], device=torch_device)

    backend = TorchTensorRTAotBackend()
    backend = backend.build(model, graph_spec, samples, device=torch_device, cache_dir=tmp_path)

    state_dict = backend.to_dict()

    torch.save(state_dict, tmp_path / "state_dict.pth")
    loaded_backend = TorchTensorRTAotBackend.from_dict(model, torch_load_with_custom_types(tmp_path / "state_dict.pth"))

    loaded_backend.activate()
    args, kwargs = samples[0]
    torch.testing.assert_close(backend.infer(*args, **kwargs), loaded_backend.infer(*args, **kwargs))


# --- TorchTensorRTAotBackendConfig.from_dict ---


def test_tensorrt_aot_config_from_dict_defaults():
    config = TorchTensorRTAotBackendConfig.from_dict({})
    default = TorchTensorRTAotBackendConfig()
    assert config.pickle_protocol == default.pickle_protocol
    assert config.pickle_protocol == 5


def test_tensorrt_aot_config_from_dict_nested_compile_config_dict():
    import aitune.torch.backend.torch_tensorrt_aot_backend as _mod

    data = {"compile_config": {"workspace_size": 2048}}
    config = TorchTensorRTAotBackendConfig.from_dict(data)
    assert isinstance(config.compile_config, _mod.TorchTensorRTConfig)
    assert config.compile_config.workspace_size == 2048


def test_tensorrt_aot_config_from_dict_compile_config_instance_passthrough():
    config_instance = TorchTensorRTAotBackendConfig().compile_config
    config = TorchTensorRTAotBackendConfig.from_dict({"compile_config": config_instance})
    assert config.compile_config is config_instance
