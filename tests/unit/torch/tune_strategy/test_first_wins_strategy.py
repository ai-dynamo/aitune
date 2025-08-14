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

"""Unit tests for FirstWinsStrategy."""

from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

from aitune.torch import Module, tune
from aitune.torch.backend import Backend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.tune_strategy.first_wins_strategy import FirstWinsStrategy
from tests.toy_backends import SleepBackend
from tests.toy_models.torch_models import ToyTorchModel


@pytest.fixture
def mock_backend():
    """Create a mock backend for testing."""
    backend = MagicMock(spec=Backend)
    backend.name = "mock_backend"
    backend.describe.return_value = "mock_backend"
    return backend


@pytest.fixture
def mock_module():
    """Create a mock torch module for testing."""
    module = MagicMock(spec=nn.Module)
    module.to = MagicMock()
    return module


@pytest.fixture
def mock_graph_spec():
    """Create a mock graph spec for testing."""
    graph_spec = MagicMock(spec=GraphSpec)
    graph_spec.name = "mock_graph_spec"
    return graph_spec


@pytest.fixture
def mock_sample():
    """Create a mock sample for testing."""
    return MagicMock(spec=Sample)


def test_describe(mock_backend):
    """Test describe method."""
    backends = [mock_backend, mock_backend]
    strategy = FirstWinsStrategy(backends)
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    assert (
        strategy.describe()
        == "name: First Wins Strategy\ndescription: evaluate backends in order, return first working backend\nbackends:\n  mock_backend\n  mock_backend"
    )


def test_tune_success_first_backend(mock_backend, mock_module, mock_graph_spec, mock_sample, torch_device, tmp_path):
    """Test tune method when first backend succeeds."""
    # Setup
    first_backend = MagicMock(spec=Backend)
    first_backend.name = "first_backend"
    first_backend.describe.return_value = "mock_backend"
    first_backend.build.return_value = mock_backend

    second_backend = MagicMock(spec=Backend)
    second_backend.name = "second_backend"
    second_backend.describe.return_value = "mock_backend"
    second_backend.build.return_value = mock_backend

    backends = [first_backend, second_backend]
    strategy = FirstWinsStrategy(backends)
    strategy._describe = MagicMock()
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    mock_backend.build.return_value = mock_backend

    # Execute
    with patch("copy.deepcopy", return_value=mock_backend):
        result = strategy.tune(mock_module, "test_module", mock_graph_spec, [mock_sample], torch_device, tmp_path)

    # Verify
    assert result == mock_backend
    mock_backend.build.assert_called_once_with(mock_module, mock_graph_spec, [mock_sample], torch_device, tmp_path)
    backends[1].build.assert_not_called()


def test_tune_success_second_backend(mock_backend, mock_module, mock_graph_spec, mock_sample, torch_device, tmp_path):
    """Test tune method when first backend fails but second succeeds."""
    # Setup
    first_backend = MagicMock(spec=Backend)
    first_backend.name = "first_backend"
    first_backend.describe.return_value = "mock_backend"
    first_backend.build.side_effect = Exception("First backend failed")

    second_backend = MagicMock(spec=Backend)
    second_backend.name = "second_backend"
    second_backend.describe.return_value = "mock_backend"
    second_backend.build.return_value = mock_backend

    backends = [first_backend, second_backend]
    strategy = FirstWinsStrategy(backends)
    strategy._describe = MagicMock()
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)

    # Execute
    with patch("copy.deepcopy", side_effect=[first_backend, second_backend]):
        result = strategy.tune(mock_module, "test_module", mock_graph_spec, [mock_sample], torch_device, tmp_path)

    # Verify
    assert result == mock_backend
    first_backend.build.assert_called_once()
    second_backend.build.assert_called_once()
    mock_module.to.assert_called_once_with(torch_device)


def test_tune_all_backends_fail(mock_module, mock_graph_spec, mock_sample, torch_device, tmp_path):
    """Test tune method when all backends fail."""
    # Setup
    backend1 = MagicMock(spec=Backend)
    backend1.name = "backend1"  # Explicitly set name as string
    backend2 = MagicMock(spec=Backend)
    backend2.name = "backend2"  # Explicitly set name as string
    backends = [backend1, backend2]

    for backend in backends:
        backend.build.side_effect = Exception("Backend failed")

    strategy = FirstWinsStrategy(backends)
    strategy._describe = MagicMock()
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)

    # Execute and verify
    with pytest.raises(RuntimeError) as exc_info:
        strategy.tune(mock_module, "test_module", mock_graph_spec, [mock_sample], torch_device, tmp_path)

    expected_error = f"There is no valid backend for a module: test_module, graph_spec: {mock_graph_spec}"
    assert str(exc_info.value) == expected_error
    assert mock_module.to.call_count == 2  # Called after each failed backend


class InferenceFailsBackend(SleepBackend):
    """Backend that fails on inference."""

    def _infer(self, *args, **kwargs):
        raise RuntimeError("Inference failed")


def test_first_wins_strategy_find_max_batch_size_fails(torch_device):
    """If find max batch size fails, there is no recovery."""
    strategy = FirstWinsStrategy(
        backends=[
            # BuildFailsBackend(RuntimeError),
            SleepBackend(),
            InferenceFailsBackend(),  # this should not be considered, as SleepBackend should be selected
        ],
    )
    strategy.enable_find_max_batch_size(True)
    strategy.enable_correctness_check(True)

    model = ToyTorchModel().eval().to(torch_device)
    sample = model.sample().to(torch_device)
    assert model(sample) is not None

    model = Module(model, strategy=strategy)
    tune(model, sample, device=torch_device, disable_external_logging=False)

    assert model(sample.unsqueeze(0)) is not None  # FIXME: does not work without unsqueeze
