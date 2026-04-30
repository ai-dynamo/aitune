# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the JIT Config contract."""

from aitune.torch.jit.config import Config
from aitune.torch.tune_strategy.first_wins_strategy import FirstWinsStrategy
from aitune.torch.tune_strategy.tune_strategy import DummyTuneStrategy


def test_resolve_strategy_default_is_first_wins():
    cfg = Config()

    strategy = cfg.resolve_strategy()

    assert isinstance(strategy, FirstWinsStrategy)
    assert len(strategy._backends) > 0  # has the built-in default backends


def test_resolve_strategy_returns_explicit_strategy_when_set():
    cfg = Config()
    explicit = DummyTuneStrategy()
    cfg.strategy = explicit

    assert cfg.resolve_strategy() is explicit
