# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OneBackendStrategy."""

from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

from aitune.torch.backend.backend import Backend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy

_PATCH_FIND_MAX_THROUGHPUT = (
    "aitune.torch.tune_strategy.mixin.performance_validation_mixin.find_max_throughput_for_backend"
)


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
    strategy = OneBackendStrategy(mock_backend)
    strategy._describe = MagicMock()
    strategy._pre_tune = MagicMock()
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    mock_backend.__deepcopy__ = lambda _, memo=None: mock_backend
    mock_backend.build.return_value = mock_backend

    # Execute
    result = strategy.tune(mock_module, "test_module", mock_graph_spec, [mock_sample], torch_device, tmp_path)

    # Verify
    assert result == mock_backend
    mock_backend.build.assert_called_once()


def test_tune_backend_fails(mock_backend, mock_module, mock_graph_spec, torch_device, mock_sample, tmp_path):
    """Test tune method when backend fails."""
    # Setup
    strategy = OneBackendStrategy(mock_backend)
    strategy._describe = MagicMock()
    strategy._pre_tune = MagicMock()
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    mock_backend.__deepcopy__ = lambda _, memo=None: mock_backend
    mock_backend.build.side_effect = Exception("Backend failed")

    # Execute and verify
    with pytest.raises(Exception) as exc_info:
        strategy.tune(mock_module, "test_module", mock_graph_spec, [mock_sample], torch_device, tmp_path)

    assert str(exc_info.value) == "Backend failed"
    mock_backend.build.assert_called_once()


def test_one_backend_falls_back_to_torch_eager_on_perf_failure(
    mock_backend, mock_module, mock_graph_spec, torch_device, mock_sample, tmp_path
):
    """When backend fails performance gate, falls back to the TorchEager baseline."""
    sink = MagicMock()
    strategy = OneBackendStrategy(mock_backend)
    strategy._sink = sink
    strategy._describe = MagicMock()
    strategy._pre_tune = MagicMock()
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    mock_backend.__deepcopy__ = lambda _, memo=None: mock_backend
    mock_backend.build.return_value = mock_backend

    eager_fallback = MagicMock(spec=Backend)
    eager_fallback.describe.return_value = "TorchEagerBackend"
    strategy._baseline_throughput = 3.0  # baseline: 3 samples/s
    strategy._resolved_batch_size = 4
    strategy._baseline_backend = eager_fallback

    # candidate is 3× slower → speedup 0.33 → fails gate
    with patch(_PATCH_FIND_MAX_THROUGHPUT, return_value=(4, 1.0, MagicMock())):
        result = strategy.tune(mock_module, "test_module", mock_graph_spec, [mock_sample], torch_device, tmp_path)

    assert result is eager_fallback
    assert len(strategy.perf_validation_results) == 1
    assert strategy.perf_validation_results[0].passed is False
    assert any("Baseline was selected" in str(call) for call in sink.call_args_list)


def test_one_backend_returns_backend_when_perf_passes(
    mock_backend, mock_module, mock_graph_spec, torch_device, mock_sample, tmp_path
):
    """When backend passes performance gate, returns the built backend."""
    strategy = OneBackendStrategy(mock_backend)
    strategy._describe = MagicMock()
    strategy._pre_tune = MagicMock()
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    mock_backend.__deepcopy__ = lambda _, memo=None: mock_backend
    mock_backend.build.return_value = mock_backend

    eager_fallback = MagicMock(spec=Backend)
    strategy._baseline_throughput = 1.0  # baseline: 1 sample/s
    strategy._resolved_batch_size = 4
    strategy._baseline_backend = eager_fallback

    # candidate is 2× faster → speedup 2.0 → passes gate
    with patch(_PATCH_FIND_MAX_THROUGHPUT, return_value=(4, 2.0, MagicMock())):
        result = strategy.tune(mock_module, "test_module", mock_graph_spec, [mock_sample], torch_device, tmp_path)

    assert result is mock_backend
    assert strategy.perf_validation_results[0].passed is True


def test_one_backend_returns_slow_backend_when_gate_disabled(
    mock_backend, mock_module, mock_graph_spec, torch_device, mock_sample, tmp_path
):
    """When validate_against_baseline=False, slow backend is returned directly (not TorchEager)."""
    strategy = OneBackendStrategy(mock_backend)
    strategy.enable_validate_against_baseline(False)
    strategy._describe = MagicMock()
    strategy._pre_tune = MagicMock()
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    mock_backend.__deepcopy__ = lambda _, memo=None: mock_backend
    mock_backend.build.return_value = mock_backend

    eager_fallback = MagicMock(spec=Backend)
    strategy._baseline_throughput = 3.0
    strategy._resolved_batch_size = 4
    strategy._baseline_backend = eager_fallback

    # 3× slower but gate is disabled
    with patch(_PATCH_FIND_MAX_THROUGHPUT, return_value=(4, 1.0, MagicMock())):
        result = strategy.tune(mock_module, "test_module", mock_graph_spec, [mock_sample], torch_device, tmp_path)

    assert result is mock_backend  # gate disabled → slow backend returned directly
    assert strategy.perf_validation_results[0].passed is False  # result still recorded
