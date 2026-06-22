# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-module strategy builder used by JIT tuning."""

from unittest.mock import Mock

import pytest

from aitune.torch.jit.config import config
from aitune.torch.jit.patched_module import _build_strategy
from aitune.torch.tune_strategy.first_wins_strategy import FirstWinsStrategy
from aitune.torch.tune_strategy.max_throughput_strategy import MaxThroughputStrategy
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy
from aitune.torch.tune_strategy.tune_strategy import DummyTuneStrategy


@pytest.fixture(autouse=True)
def reset_jit_config():
    """Reset JIT config to defaults so changes are scoped to the test."""
    yield
    config.reset_to_defaults()


def test_build_strategy_default_is_first_wins_with_find_max_batch_size_disabled():
    # No override; resolve_strategy() builds the default FirstWinsStrategy.
    strategy = _build_strategy()

    assert isinstance(strategy, FirstWinsStrategy)
    assert strategy._enable_find_max_batch_size is False


def test_build_strategy_uses_configured_strategy():
    user_strategy = OneBackendStrategy(backend=Mock(name="user_backend"))
    config.strategy = user_strategy

    strategy = _build_strategy()

    assert isinstance(strategy, OneBackendStrategy)
    assert strategy._enable_find_max_batch_size is False


def test_build_strategy_clones_per_call_so_state_is_isolated():
    config.strategy = OneBackendStrategy(backend=Mock(name="user_backend"))

    first = _build_strategy()
    second = _build_strategy()

    assert first is not config.strategy
    assert second is not config.strategy
    assert first is not second


def test_build_strategy_disables_find_max_batch_size_for_max_throughput():
    config.strategy = MaxThroughputStrategy(backends=[Mock(name="backend")])

    strategy = _build_strategy()

    assert isinstance(strategy, MaxThroughputStrategy)
    assert strategy._enable_find_max_batch_size is False


def test_build_strategy_handles_strategy_without_find_max_batch_size_extension():
    """Custom strategies that don't subclass the find-max-batch-size extension still work."""
    config.strategy = DummyTuneStrategy()

    strategy = _build_strategy()

    assert isinstance(strategy, DummyTuneStrategy)
