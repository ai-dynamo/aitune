# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FirstWinsStrategy."""

from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from aitune.torch import Module, tune
from aitune.torch.backend import Backend
from aitune.torch.module.graph_spec import GraphSpec
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
def mock_sample():
    """Create a mock sample for testing."""
    return ["this is a mock sample"]


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
    first_backend.__deepcopy__ = lambda _: first_backend
    first_backend.build.return_value = mock_backend

    second_backend = MagicMock(spec=Backend)
    second_backend.name = "second_backend"
    second_backend.describe.return_value = "mock_backend"
    second_backend.__deepcopy__ = lambda _: second_backend
    second_backend.build.return_value = mock_backend

    backends = [first_backend, second_backend]
    strategy = FirstWinsStrategy(backends)
    strategy._describe = MagicMock()
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)

    # Execute
    result = strategy.tune(mock_module, "test_module", mock_graph_spec, [mock_sample], torch_device, tmp_path)

    # Verify
    assert result == mock_backend
    backends[0].build.assert_called_once()
    backends[1].build.assert_not_called()


def test_tune_success_second_backend(mock_backend, mock_module, mock_graph_spec, mock_sample, torch_device, tmp_path):
    """Test tune method when first backend fails but second succeeds."""
    # Setup
    first_backend = MagicMock(spec=Backend)
    first_backend.name = "first_backend"
    first_backend.describe.return_value = "mock_backend"
    first_backend.__deepcopy__ = lambda _: first_backend
    first_backend.build.side_effect = Exception("First backend failed")

    second_backend = MagicMock(spec=Backend)
    second_backend.name = "second_backend"
    second_backend.describe.return_value = "mock_backend"
    second_backend.__deepcopy__ = lambda _: second_backend
    second_backend.build.return_value = mock_backend

    backends = [first_backend, second_backend]
    strategy = FirstWinsStrategy(backends)
    strategy._describe = MagicMock()
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)

    # Execute
    result = strategy.tune(mock_module, "test_module", mock_graph_spec, mock_sample, torch_device, tmp_path)

    # Verify
    assert result == mock_backend
    first_backend.build.assert_called_once()
    second_backend.build.assert_called_once()
    mock_module.to.assert_called_once_with(torch_device)

    # Ensure that the data passed to build() is a different object than the original [mock_sample]
    # That is, each backend gets its own copy of the sample, not the original list, nor the same object
    for backend in backends:
        build_data_args = [args[2] for args, kwargs in backend.build.call_args_list]
        assert build_data_args[0] is not mock_sample  # different list instance


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
