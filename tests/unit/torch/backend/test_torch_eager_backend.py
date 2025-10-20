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
"""Unit tests for TorchInductorBackend."""

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from aitune.torch.backend.torch_eager import TorchEagerBackend, TorchEagerBackendConfig
from aitune.torch.checkpoint.storage_tasks import TorchLoadTask, TorchSaveTask
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.module.sample_metadata import SampleMetadata
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


@pytest.fixture
def sample_metadata(sample_data) -> SampleMetadata:
    return SampleMetadata(sample_data[0])


def move_to_dtype(sample_data, dtype):
    args, kwargs = sample_data[0]
    args = (args[0].to(dtype),)
    return [(args, kwargs)]


def backend_build(backend, dtype, model, sample_data, tmp_path, device="cuda"):
    """Build the model with the backend."""
    device = torch.device(device)
    model = model.to(device, dtype=dtype)
    data = move_to_dtype(sample_data, dtype)
    backend = backend.build(model, graph_spec=Mock(spec=GraphSpec), data=data, device=device, cache_dir=tmp_path)
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
