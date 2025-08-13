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

"""Unit tests for OneBackendStrategy."""

from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

from aitune.torch.backend.backend import Backend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy


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
    """Test describe method returns correct string."""
    strategy = OneBackendStrategy(mock_backend)
    assert strategy.describe() == "name: One Backend Strategy\ndescription: Use only one backend\nbackend: mock_backend"


def test_tune_success(mock_backend, mock_module, mock_graph_spec, torch_device, mock_sample, tmp_path):
    """Test tune method when backend succeeds."""
    # Setup
    strategy = OneBackendStrategy(MagicMock(spec=Backend))
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


def test_tune_backend_fails(mock_backend, mock_module, mock_graph_spec, torch_device, mock_sample, tmp_path):
    """Test tune method when backend fails."""
    # Setup
    strategy = OneBackendStrategy(MagicMock(spec=Backend))
    strategy._describe = MagicMock()
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    mock_backend.build.side_effect = Exception("Backend failed")

    # Execute and verify
    with pytest.raises(Exception) as exc_info, patch("copy.deepcopy", return_value=mock_backend):
        strategy.tune(mock_module, "test_module", mock_graph_spec, [mock_sample], torch_device, tmp_path)

    assert str(exc_info.value) == "Backend failed"
    mock_backend.build.assert_called_once_with(mock_module, mock_graph_spec, [mock_sample], torch_device, tmp_path)
