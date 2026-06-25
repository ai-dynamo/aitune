# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for tune strategy mixin composition."""

from aitune.torch.tune_strategy.first_wins_strategy import FirstWinsStrategy
from aitune.torch.tune_strategy.latency_budget_strategy import LatencyBudgetStrategy
from aitune.torch.tune_strategy.max_throughput_strategy import MaxThroughputStrategy
from aitune.torch.tune_strategy.mixin import FindMaxBatchSizeMixin, PerformanceValidationMixin
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy
from aitune.torch.tune_strategy.tune_strategy import TuneStrategy


def test_strategy_mixins_are_tune_strategy_base_classes():
    assert issubclass(FindMaxBatchSizeMixin, TuneStrategy)
    assert issubclass(PerformanceValidationMixin, TuneStrategy)
    assert not issubclass(PerformanceValidationMixin, FindMaxBatchSizeMixin)
    assert issubclass(FirstWinsStrategy, PerformanceValidationMixin)
    assert issubclass(FirstWinsStrategy, FindMaxBatchSizeMixin)
    assert issubclass(FirstWinsStrategy, TuneStrategy)
    assert issubclass(OneBackendStrategy, PerformanceValidationMixin)
    assert issubclass(OneBackendStrategy, FindMaxBatchSizeMixin)
    assert issubclass(OneBackendStrategy, TuneStrategy)
    assert issubclass(LatencyBudgetStrategy, FindMaxBatchSizeMixin)
    assert issubclass(LatencyBudgetStrategy, TuneStrategy)
    assert issubclass(MaxThroughputStrategy, FindMaxBatchSizeMixin)
    assert issubclass(MaxThroughputStrategy, TuneStrategy)


def test_strategy_mro_orders_colliding_hooks():
    first_wins_mro = FirstWinsStrategy.__mro__
    one_backend_mro = OneBackendStrategy.__mro__
    latency_budget_mro = LatencyBudgetStrategy.__mro__
    max_throughput_mro = MaxThroughputStrategy.__mro__

    assert first_wins_mro.index(PerformanceValidationMixin) < first_wins_mro.index(FindMaxBatchSizeMixin)
    assert first_wins_mro.index(FindMaxBatchSizeMixin) < first_wins_mro.index(TuneStrategy)
    assert first_wins_mro.count(TuneStrategy) == 1
    assert one_backend_mro.index(PerformanceValidationMixin) < one_backend_mro.index(FindMaxBatchSizeMixin)
    assert one_backend_mro.index(FindMaxBatchSizeMixin) < one_backend_mro.index(TuneStrategy)
    assert one_backend_mro.count(TuneStrategy) == 1
    assert latency_budget_mro.index(FindMaxBatchSizeMixin) < latency_budget_mro.index(TuneStrategy)
    assert latency_budget_mro.count(TuneStrategy) == 1
    assert max_throughput_mro.index(FindMaxBatchSizeMixin) < max_throughput_mro.index(TuneStrategy)
    assert max_throughput_mro.count(TuneStrategy) == 1
