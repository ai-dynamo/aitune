# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FirstWinsStrategy."""

from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from aitune.torch import Module, tune
from aitune.torch.backend import Backend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import SampleStore
from aitune.torch.module.wrapper_module import ModuleState
from aitune.torch.tune_strategy.first_wins_strategy import FirstWinsStrategy
from tests.toy_backends import BuildFailsBackend, SleepBackend
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
def mock_samples():
    """Create a mock sample store for testing."""
    return MagicMock(spec=SampleStore)


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


def test_tune_success_first_backend(mock_backend, mock_module, mock_graph_spec, mock_samples, torch_device, tmp_path):
    """Test tune method when first backend succeeds."""
    # Setup
    first_backend = MagicMock(spec=Backend)
    first_backend.name = "first_backend"
    first_backend.describe.return_value = "mock_backend"
    first_backend.__deepcopy__ = lambda _, memo=None: first_backend
    first_backend.build.return_value = mock_backend

    second_backend = MagicMock(spec=Backend)
    second_backend.name = "second_backend"
    second_backend.describe.return_value = "mock_backend"
    second_backend.__deepcopy__ = lambda _, memo=None: second_backend
    second_backend.build.return_value = mock_backend

    backends = [first_backend, second_backend]
    strategy = FirstWinsStrategy(backends)
    strategy._describe = MagicMock()
    strategy._pre_tune = MagicMock()
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)

    # Execute
    result = strategy.tune(mock_module, "test_module", mock_graph_spec, mock_samples, torch_device, tmp_path)

    # Verify
    assert result == mock_backend
    backends[0].build.assert_called_once()
    backends[1].build.assert_not_called()


def test_tune_success_second_backend(mock_backend, mock_module, mock_graph_spec, mock_samples, torch_device, tmp_path):
    """Test tune method when first backend fails but second succeeds."""
    # Setup
    first_backend = MagicMock(spec=Backend)
    first_backend.name = "first_backend"
    first_backend.describe.return_value = "mock_backend"
    first_backend.__deepcopy__ = lambda _, memo=None: first_backend
    first_backend.build.side_effect = Exception("First backend failed")

    second_backend = MagicMock(spec=Backend)
    second_backend.name = "second_backend"
    second_backend.describe.return_value = "mock_backend"
    second_backend.__deepcopy__ = lambda _, memo=None: second_backend
    second_backend.build.return_value = mock_backend

    backends = [first_backend, second_backend]
    strategy = FirstWinsStrategy(backends)
    strategy._describe = MagicMock()
    strategy._pre_tune = MagicMock()
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)

    # Execute
    result = strategy.tune(mock_module, "test_module", mock_graph_spec, mock_samples, torch_device, tmp_path)

    # Verify
    assert result == mock_backend
    first_backend.build.assert_called_once()
    second_backend.build.assert_called_once()
    mock_module.to.assert_called_once_with(torch_device)

    # Each backend shares the disk-backed store and loads transient samples as needed.
    for backend in backends:
        build_samples = [args[2] for args, kwargs in backend.build.call_args_list]
        assert build_samples[0] is mock_samples


def test_tune_all_backends_fail(mock_module, mock_graph_spec, mock_samples, torch_device, tmp_path):
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
    strategy._pre_tune = MagicMock()
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)

    # Execute and verify
    with pytest.raises(RuntimeError) as exc_info:
        strategy.tune(mock_module, "test_module", mock_graph_spec, mock_samples, torch_device, tmp_path)

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
            SleepBackend(),
        ],
    )
    strategy.enable_find_max_batch_size(True)
    strategy.set_find_max_batch_size_default_backend_class(InferenceFailsBackend)

    model = ToyTorchModel().eval().to(torch_device)
    sample = model.sample().to(torch_device)
    assert model(sample) is not None

    model = Module(model, strategy=strategy)
    tune(model, sample, device=torch_device, disable_external_logging=False)

    assert model.state == ModuleState.PASSTHROUGH


def test_first_wins_strategy_build_fails(torch_device, tmp_path):
    class TestOutOfMemoryException(Exception):
        """Test out of memory exception."""

    strategy = FirstWinsStrategy(
        backends=[
            BuildFailsBackend(TestOutOfMemoryException),
        ],
    )
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)

    model = ToyTorchModel().eval().to(torch_device)
    sample = model.sample().to(torch_device)

    model = Module(model, strategy=strategy)

    tune(model, sample, device=torch_device, disable_external_logging=False)

    assert model.state == ModuleState.PASSTHROUGH

    # Check if TestOutOfMemoryException is mentioned in the build log
    log_files = list(tmp_path.rglob("build.log"))
    assert len(log_files) == 1, "No build.log file found in tmp_path"
    log_file = log_files[0]

    assert "TestOutOfMemoryException" in log_file.read_text()


def test_first_wins_skips_slow_backend(
    mock_backend, mock_module, mock_graph_spec, mock_samples, torch_device, tmp_path
):
    """A backend slower than TorchEager baseline by >threshold is skipped."""
    from unittest.mock import patch

    slow_backend = MagicMock(spec=Backend)
    slow_backend.describe.return_value = "slow_backend"
    slow_backend.key.return_value = "slow_backend"
    slow_backend.__deepcopy__ = lambda _, memo=None: slow_backend
    slow_backend.build.return_value = slow_backend

    fast_backend = MagicMock(spec=Backend)
    fast_backend.describe.return_value = "fast_backend"
    fast_backend.key.return_value = "fast_backend"
    fast_backend.__deepcopy__ = lambda _, memo=None: fast_backend
    fast_backend.build.return_value = fast_backend

    strategy = FirstWinsStrategy([slow_backend, fast_backend])
    strategy._describe = MagicMock()
    strategy._pre_tune = MagicMock()
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    strategy._baseline_throughput = 1.0  # baseline: 1 sample/s
    strategy._resolved_batch_size = 4

    baseline_eager = MagicMock(spec=Backend)
    baseline_eager.describe.return_value = "TorchEager"
    strategy._baseline_backend = baseline_eager

    # slow_backend: 0.5 samples/s (speedup 0.5, fails), fast_backend: 2.0 samples/s (speedup 2.0, passes)
    with patch(
        "aitune.torch.tune_strategy.mixin.performance_validation_mixin.find_max_throughput_for_backend",
        side_effect=[(4, 0.5, MagicMock()), (4, 2.0, MagicMock())],
    ):
        result = strategy.tune(mock_module, "test_module", mock_graph_spec, mock_samples, torch_device, tmp_path)

    assert result is fast_backend
    slow_backend.build.assert_called_once()
    fast_backend.build.assert_called_once()

    assert len(strategy.perf_validation_results) == 2
    assert strategy.perf_validation_results[0].passed is False  # slow
    assert strategy.perf_validation_results[1].passed is True  # fast


def test_first_wins_skips_perf_profiling_when_validation_disabled(
    mock_backend, mock_module, mock_graph_spec, mock_samples, torch_device, tmp_path
):
    """When performance validation is disabled, the first correct backend is returned without profiling."""
    from unittest.mock import patch

    slow_backend = MagicMock(spec=Backend)
    slow_backend.describe.return_value = "slow_backend"
    slow_backend.key.return_value = "slow_backend"
    slow_backend.__deepcopy__ = lambda _, memo=None: slow_backend
    slow_backend.build.return_value = slow_backend

    strategy = FirstWinsStrategy([slow_backend])
    strategy.enable_performance_validation(False)
    strategy._describe = MagicMock()
    strategy._pre_tune = MagicMock()
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    strategy._baseline_throughput = 2.0
    strategy._resolved_batch_size = 4

    baseline_eager = MagicMock(spec=Backend)
    strategy._baseline_backend = baseline_eager

    with patch(
        "aitune.torch.tune_strategy.mixin.performance_validation_mixin.find_max_throughput_for_backend",
    ) as mock_profile:
        result = strategy.tune(mock_module, "test_module", mock_graph_spec, mock_samples, torch_device, tmp_path)

    assert result is slow_backend
    mock_profile.assert_not_called()
    assert strategy.perf_validation_results == []
