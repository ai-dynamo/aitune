# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import cast
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from aitune.exceptions import AITuneError
from aitune.torch.backend.torch_tensorrt_jit_backend import (
    TorchTensorRTJitBackend,
    TorchTensorRTJitBackendConfig,
)
from aitune.torch.checkpoint.storage_tasks import torch_load_with_custom_types
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.module.sample_metadata import SampleMetadata
from tests.toy_models import ToyTorchModel
from tests.utilities.helpers import requires_cuda


@dataclass
class TorchTensorRTTestConfig:
    enabled_precisions: set[torch.dtype] = field(default_factory=lambda: {torch.float16})
    workspace_size: int = 0


@pytest.fixture
def model(torch_device) -> nn.Module:
    return ToyTorchModel().to(torch_device).eval()


@pytest.fixture
def sample_data(model, torch_device) -> list[Sample]:
    return model.samples(device=torch_device)


@pytest.fixture
def sample_metadata(sample_data) -> SampleMetadata:
    return SampleMetadata(sample_data[0])


@pytest.fixture
def torch_tensorrt_jit_backend_config() -> TorchTensorRTJitBackendConfig:
    return TorchTensorRTJitBackendConfig(
        compile_config=TorchTensorRTTestConfig(
            enabled_precisions={torch.float16},
            workspace_size=1,
        ),
        dynamic_shapes=False,
    )


@pytest.fixture
def torch_tensorrt_jit_backend(torch_tensorrt_jit_backend_config, torch_device, mocker) -> TorchTensorRTJitBackend:
    mocker.patch("aitune.torch.backend.torch_tensorrt_jit_backend.assert_cuda_is_available")
    mocker.patch("aitune.torch.backend.torch_tensorrt_jit_backend.assert_torch_tensorrt")

    return TorchTensorRTJitBackend(torch_tensorrt_jit_backend_config)


def backend_build(torch_tensorrt_jit_backend, model, sample_data, torch_device, tmp_path, mocker):
    torch_mod = mocker.patch("aitune.torch.backend.torch_tensorrt_jit_backend.torch")
    torch_mod.compile = mocker.MagicMock(return_value=model)

    backend = torch_tensorrt_jit_backend.build(
        model, graph_spec=Mock(spec=GraphSpec), data=sample_data, device=torch_device, cache_dir=tmp_path
    )
    backend = cast(TorchTensorRTJitBackend, backend)
    return torch_mod, backend


def test_torch_tensorrt_jit_backend_config_key():
    """Test backend config with cache_dir."""
    config = TorchTensorRTJitBackendConfig()
    key1 = config.key()
    key2 = config.key()

    assert key1 == key2


def test_torch_tensorrt_jit_backend_config_describe():
    """Test backend config describe."""
    config = TorchTensorRTJitBackendConfig(compile_config=TorchTensorRTTestConfig())
    describe = config.describe()

    assert describe == "compile_config=TorchTensorRTConfig(enabled_precisions={torch.float16})"

    config = TorchTensorRTJitBackendConfig(compile_config=TorchTensorRTTestConfig(workspace_size=1))
    describe = config.describe()

    assert describe == "compile_config=TorchTensorRTConfig(enabled_precisions={torch.float16},workspace_size=1)"

    config = TorchTensorRTJitBackendConfig(compile_config=TorchTensorRTTestConfig(), fullgraph=True)
    describe = config.describe()

    assert describe == "compile_config=TorchTensorRTConfig(enabled_precisions={torch.float16}),fullgraph=True"


def test_mock_build(torch_tensorrt_jit_backend, model, sample_data, torch_device, tmp_path, mocker):
    torch_mod = mocker.patch("aitune.torch.backend.torch_tensorrt_jit_backend.torch")
    torch_mod.compile = mocker.MagicMock(return_value=model)

    backend = torch_tensorrt_jit_backend.build(
        model, graph_spec=Mock(spec=GraphSpec), data=sample_data, device=torch_device, cache_dir=tmp_path
    )

    assert backend is torch_tensorrt_jit_backend
    assert backend._compiled_module is not None


def test_mock_infer(torch_tensorrt_jit_backend, model, sample_data, torch_device, tmp_path, mocker):
    torch_mod, backend = backend_build(torch_tensorrt_jit_backend, model, sample_data, torch_device, tmp_path, mocker)

    assert backend._compiled_module is not None

    # Get sample input
    args, kwargs = sample_data[0]

    # Test inference
    output = backend.infer(*args, **kwargs)
    assert output is not None

    # Verify the mock was called with the right arguments
    torch_mod.compile.assert_called_once()

    assert torch_mod.compile.call_args[1]["backend"] == "torch_tensorrt"

    #     mocker.ANY,
    #     backend=
    #     options=asdict(torch_tensorrt_jit_backend_config.compile_config),
    #     dynamic=torch_tensorrt_jit_backend_config.dynamic_shapes,
    # )

    # Reset for cleanup
    backend.deactivate()
    assert backend._compiled_module is None


def test_torch_compile_backend_infer_not_activated(
    torch_tensorrt_jit_backend, model, sample_data, torch_device, tmp_path, mocker
):
    torch_mod = mocker.patch("aitune.torch.backend.torch_tensorrt_jit_backend.torch")
    torch_mod.compile = mocker.MagicMock(return_value=model)
    torch_mod._dynamo.reset = mocker.MagicMock()
    torch_mod.cuda.empty_cache = mocker.MagicMock()
    torch_mod.collect = mocker.MagicMock()

    backend = torch_tensorrt_jit_backend.build(
        model, graph_spec=Mock(spec=GraphSpec), data=sample_data, device=torch_device, cache_dir=tmp_path
    )
    backend = cast(TorchTensorRTJitBackend, backend)

    backend._compiled_module = None

    with pytest.raises(AITuneError):
        backend.infer(torch.randn(2, 256))


def test_torch_compile_backend_deactivate(
    torch_tensorrt_jit_backend, model, sample_data, torch_device, tmp_path, mocker
):
    torch_mod = mocker.patch("aitune.torch.backend.torch_tensorrt_jit_backend.torch")
    torch_mod.compile = mocker.MagicMock(return_value=model)
    torch_mod._dynamo.reset = mocker.MagicMock()
    torch_mod.cuda.empty_cache = mocker.MagicMock()
    torch_mod.collect = mocker.MagicMock()

    mock_gc = mocker.patch("gc.collect")

    # Build, activate, and deactivate
    backend = torch_tensorrt_jit_backend.build(
        model, graph_spec=Mock(spec=GraphSpec), data=sample_data, device=torch_device, cache_dir=tmp_path
    )
    backend.activate()
    backend.deactivate()

    # Verify cleanup was performed
    torch_mod._dynamo.reset.assert_called()
    torch_mod.cuda.empty_cache.assert_called()
    mock_gc.assert_called()


@requires_cuda
def test_torch_compile_backend_init(torch_tensorrt_jit_backend_config):
    backend = TorchTensorRTJitBackend(torch_tensorrt_jit_backend_config)
    assert backend._config == torch_tensorrt_jit_backend_config


@requires_cuda
def test_torch_compile_backend_build(torch_tensorrt_jit_backend, model, sample_data, torch_device, tmp_path):
    backend = torch_tensorrt_jit_backend.build(
        model, graph_spec=Mock(spec=GraphSpec), data=sample_data, device=torch_device, cache_dir=tmp_path
    )
    assert backend is torch_tensorrt_jit_backend
    assert backend._orig_module is model
    assert backend._compiled_module is not None


@requires_cuda
def test_torch_compile_backend_compile(torch_tensorrt_jit_backend, model, sample_data, torch_device, tmp_path):
    args, kwargs = sample_data[0]
    backend = torch_tensorrt_jit_backend.build(
        model, graph_spec=Mock(spec=GraphSpec), data=sample_data, device=torch_device, cache_dir=tmp_path
    )
    backend.infer(*args, **kwargs)


@requires_cuda
def test_serialization(torch_tensorrt_jit_backend, model, sample_data, torch_device, mocker, tmp_path):
    _, backend = backend_build(torch_tensorrt_jit_backend, model, sample_data, torch_device, tmp_path, mocker)
    state_dict = backend.to_dict()

    torch.save(state_dict, tmp_path / "state_dict.pth")
    loaded_backend = TorchTensorRTJitBackend.from_dict(model, torch_load_with_custom_types(tmp_path / "state_dict.pth"))

    loaded_backend.activate()
    args, kwargs = sample_data[0]
    torch.testing.assert_close(backend.infer(*args, **kwargs), loaded_backend.infer(*args, **kwargs))
