# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
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
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.module.sample_metadata import SampleMetadata
from tests.toy_models import ToyTorchModel
from tests.toy_models.torch_models import ToyTorchConditionalModel
from tests.utilities.helpers import requires_cuda


@dataclass
class TorchTensorRTTestConfig:
    enabled_precisions: set[torch.dtype] = field(default_factory=lambda: {torch.float16})
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
    return GraphSpec(
        "0",
        input_spec=SampleMetadata.from_inputs(args, kwargs, strict=True),
        output_spec=SampleMetadata.from_outputs(output),
    )


@pytest.fixture
def backend(mocker) -> TorchTensorRTAotBackend:
    from aitune.torch.backend.torch_tensorrt_aot_backend import torch_tensorrt

    if torch_tensorrt is None:
        pytest.skip("torch_tensorrt is not available")

    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.assert_cuda_is_available")  # always available

    return TorchTensorRTAotBackend(config=TorchTensorRTAotBackendConfig(compile_config=TorchTensorRTTestConfig()))


@pytest.fixture
def mock_torch_tensorrt(mocker, model: SimpleModel):
    mock_torch_tensorrt = mocker.Mock()
    mock_torch_tensorrt.compile = mocker.Mock(return_value=model)
    mock_torch_tensorrt.save = mocker.Mock()

    mocker.patch("aitune.torch.backend.torch_tensorrt_aot_backend.torch_tensorrt", mock_torch_tensorrt)
    return mock_torch_tensorrt


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

    assert describe == "compile_config=TorchTensorRTConfig(enabled_precisions={torch.float16})"

    config = TorchTensorRTAotBackendConfig(compile_config=TorchTensorRTTestConfig(workspace_size=1))
    describe = config.describe()

    assert describe == "compile_config=TorchTensorRTConfig(enabled_precisions={torch.float16},workspace_size=1)"

    config = TorchTensorRTAotBackendConfig(compile_config=TorchTensorRTTestConfig(), pickle_protocol=1)
    describe = config.describe()

    assert describe == "compile_config=TorchTensorRTConfig(enabled_precisions={torch.float16}),pickle_protocol=1"


@requires_cuda
def test_mock_build(
    mock_torch_tensorrt,
    backend: TorchTensorRTAotBackend,
    model: SimpleModel,
    graph_spec: GraphSpec,
    sample_data: list[Sample],
    torch_device: torch.device,
    tmp_path: Path,
):
    active_backend = backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)

    assert active_backend is backend

    active_backend = cast(TorchTensorRTAotBackend, active_backend)
    assert active_backend._device == torch_device
    assert active_backend.is_active

    mock_torch_tensorrt.compile.assert_called_once()


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

    config = TorchTensorRTAotBackendConfig(
        compile_config=TorchTensorRTTestConfig(
            enabled_precisions={torch.float16},
            workspace_size=1,
        )
    )
    backend = TorchTensorRTAotBackend(
        config=config,
    )

    backend = backend.build(model, graph_spec, sample_data, device=torch_device, cache_dir=tmp_path)
    assert backend is not None

    mock_torch_tensorrt.compile.assert_called_once()
    assert mock_torch_tensorrt.compile.call_args[1]["workspace_size"] == 1


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


@pytest.fixture
def dynamic_samples_data(torch_device):
    dynamic_samples = [
        ((torch.randn(1, 4, 244, device=torch_device),), {}),
        ((torch.randn(2, 16, 244, device=torch_device),), {}),
        ((torch.randn(4, 256, 244, device=torch_device),), {}),
    ]
    return dynamic_samples


@pytest.fixture
def graph_spec_with_dynamic_shape(dynamic_samples_data, model) -> GraphSpec:
    args, kwargs = dynamic_samples_data[0]
    input_metadata = SampleMetadata.from_inputs(args, kwargs, strict=True)
    output_metadata = SampleMetadata.from_outputs(dynamic_samples_data[0])
    for sample in dynamic_samples_data[1:]:
        args, kwargs = sample
        input_metadata.update_shapes_seen(SampleMetadata.from_inputs(args, kwargs, strict=True))
        output_metadata.update_shapes_seen(SampleMetadata.from_outputs(sample))

    graph_spec = GraphSpec("0", input_metadata, output_metadata)
    return graph_spec


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
