# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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
"""Test for tune function."""

import logging
from collections import Counter
from unittest.mock import Mock

import pytest
import torch
from torch.utils.data import Dataset

from aitune.torch import tuning
from aitune.torch.config import DEFAULT_DEVICE
from aitune.torch.dataloader import DataLoaderFactory
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.tuning import LOG_FORMAT, tune


class DummyDataset(Dataset):
    def __init__(self, size=10):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randn(3, 32, 32)


@pytest.mark.parametrize("dry_run", [True, False])
def test_tune(mocker, dry_run):
    # given
    mock_module = Mock()
    mock_module.backends = {}
    dataset = DummyDataset()
    batch_sizes = [1, 2, 4]
    max_batches = 2

    mocker.patch.dict(MODULE_REGISTRY.modules, {"test_module": mock_module}, clear=True)
    mocker.patch("aitune.torch.tuning._describe_module")

    # when
    tune(
        mock_module,
        dataset,
        batch_sizes=batch_sizes,
        max_num_batches_per_batch_size=max_batches,
        dry_run=dry_run,
        disable_external_logging=False,
    )

    # then
    assert mock_module.call_count > 0
    mock_module.tune.assert_called_once_with(dry_run=dry_run, device=torch.device(DEFAULT_DEVICE))

    batch_size_counter = Counter()
    for args, _ in mock_module.call_args_list:
        bs = args[0].size(0)
        batch_size_counter[bs] += 1

    assert batch_size_counter == Counter({1: max_batches, 2: max_batches, 4: max_batches})


@pytest.mark.parametrize("clear_cache", [True, False])
def test_tune_with_cache_clear(mocker, clear_cache):
    # given
    mock_module = Mock()
    mock_module.backends = {}
    dataset = DummyDataset()
    batch_sizes = [1, 2, 4]
    max_batches = 2

    mocker.patch.dict(MODULE_REGISTRY.modules, {"test_module": mock_module}, clear=True)
    mocker.patch("aitune.torch.tuning._describe_module")
    spy_clear_cache = mocker.spy(tuning, "_clear_cache")

    # when
    tune(
        mock_module,
        dataset,
        batch_sizes=batch_sizes,
        max_num_batches_per_batch_size=max_batches,
        disable_external_logging=False,
        clear_cache=clear_cache,
    )

    # then
    assert mock_module.call_count > 0
    mock_module.tune.assert_called_once_with(dry_run=False, device=torch.device(DEFAULT_DEVICE))

    assert spy_clear_cache.call_count == (1 if clear_cache else 0)

    batch_size_counter = Counter()
    for args, _ in mock_module.call_args_list:
        bs = args[0].size(0)
        batch_size_counter[bs] += 1

    assert batch_size_counter == Counter({1: max_batches, 2: max_batches, 4: max_batches})


def test_tune_with_dataloader_factory(mocker):
    # given
    mock_module = Mock()
    mock_module.backends = {}
    dataset = DummyDataset()
    factory = DataLoaderFactory(dataset)

    mocker.patch.dict(MODULE_REGISTRY.modules, {"test_module": mock_module}, clear=True)
    mocker.patch("aitune.torch.tuning._describe_module")

    # when
    tune(mock_module, factory, batch_sizes=[2], disable_external_logging=False)

    # then
    assert mock_module.call_count > 0


def test_tune_log_level(mocker):
    """Test that the tune function correctly responds to global logging level."""
    # given
    mock_func = Mock()
    dataset = DummyDataset(size=1)  # Small dataset for quick test

    # Mock the enable_gpu_memory_logging function
    mock_enable_gpu_memory_logging = mocker.patch("aitune.utils.logging.enable_gpu_memory_logging")
    spy_setup_logging = mocker.spy(tuning, "setup_logging")

    # Get the root logger to check its level
    root_logger = logging.getLogger()
    original_level = root_logger.level

    mocker.patch.dict(MODULE_REGISTRY.modules, {}, clear=True)

    try:
        # Test 1: Global logging level DEBUG - should enable GPU memory logging
        root_logger.setLevel(logging.DEBUG)
        tune(mock_func, dataset, batch_sizes=[1])
        spy_setup_logging.assert_called_once_with(format_string=LOG_FORMAT)
        mock_enable_gpu_memory_logging.assert_called_once()

        # Reset
        mock_enable_gpu_memory_logging.reset_mock()
        spy_setup_logging.reset_mock()

        # Test 2: Global logging level WARNING - should NOT enable GPU memory logging
        root_logger.setLevel(logging.WARNING)
        tune(mock_func, dataset, batch_sizes=[1])
        spy_setup_logging.assert_called_once_with(format_string=LOG_FORMAT)
        mock_enable_gpu_memory_logging.assert_not_called()

        # Reset
        mock_enable_gpu_memory_logging.reset_mock()
        spy_setup_logging.reset_mock()

        # Test 3: Global logging level INFO - should NOT enable GPU memory logging
        root_logger.setLevel(logging.INFO)
        tune(mock_func, dataset, batch_sizes=[1])
        spy_setup_logging.assert_called_once_with(format_string=LOG_FORMAT)
        mock_enable_gpu_memory_logging.assert_not_called()

        # Reset
        mock_enable_gpu_memory_logging.reset_mock()
        spy_setup_logging.reset_mock()

        # Test 4: Global logging level ERROR - should NOT enable GPU memory logging
        root_logger.setLevel(logging.ERROR)
        tune(mock_func, dataset, batch_sizes=[1])
        spy_setup_logging.assert_called_once_with(format_string=LOG_FORMAT)
        mock_enable_gpu_memory_logging.assert_not_called()

    finally:
        # Restore original logging level
        root_logger.setLevel(original_level)


def test_tune_disable_external_logging(mocker):
    """Test that the tune function correctly responds to disable_external_logging parameter."""
    # given
    mock_func = Mock()
    dataset = DummyDataset(size=1)  # Small dataset for quick test

    spy_libraries_logging = mocker.spy(tuning, "libraries_logging")

    mocker.patch.dict(MODULE_REGISTRY.modules, {}, clear=True)

    # when
    tune(mock_func, dataset, batch_sizes=[1], disable_external_logging=False)

    # then
    spy_libraries_logging.assert_called_once_with(False)

    # Reset spy for next test
    spy_libraries_logging.reset_mock()

    # when
    tune(mock_func, dataset, batch_sizes=[1], disable_external_logging=True)

    # then
    spy_libraries_logging.assert_called_once_with(True)
