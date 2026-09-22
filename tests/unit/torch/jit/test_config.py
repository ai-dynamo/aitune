# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the JIT Config contract."""

import torch

from aitune.torch.jit.config import Config
from aitune.torch.tune_strategy.max_throughput_strategy import MaxThroughputStrategy
from aitune.torch.tune_strategy.tune_strategy import DummyTuneStrategy


def test_resolve_strategy_default_is_max_throughput():
    cfg = Config()

    strategy = cfg.resolve_strategy(torch.nn.Identity())

    assert isinstance(strategy, MaxThroughputStrategy)
    assert len(strategy._backends) > 0  # has the built-in default backends


def test_resolve_strategy_returns_explicit_strategy_when_set():
    cfg = Config()
    explicit = DummyTuneStrategy()
    cfg.strategy = explicit

    assert cfg.resolve_strategy(torch.nn.Identity()) is explicit
