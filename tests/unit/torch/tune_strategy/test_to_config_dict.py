# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TuneStrategy.to_json_dict and concrete overrides."""

from unittest.mock import MagicMock

from aitune.torch.backend.backend import Backend
from aitune.torch.tune_strategy.first_wins_strategy import FirstWinsStrategy
from aitune.torch.tune_strategy.max_throughput_strategy import MaxThroughputStrategy
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy
from aitune.torch.tune_strategy.tune_strategy import DummyTuneStrategy


def _assert_strategy_profiling_defaults(result):
    profiling_config = result["profiling_config"]
    assert profiling_config["batching"] is True
    assert profiling_config["batch_sizes"][0] == 1
    assert profiling_config["batch_sizes"][-1] == 2**20
    assert profiling_config["measuring_strategy"] == "ModelExecutionTimeMeasuringStrategy"
    assert profiling_config["measurement_stop_strategy"] == "NumStepsMeasuringStopStrategy"
    assert profiling_config["profiling_stop_strategy"] == "ThroughputSaturatedProfilingStopStrategy"


def test_dummy_strategy_to_json_dict():
    strategy = DummyTuneStrategy()
    result = strategy.to_json_dict()
    assert result == {}


def test_first_wins_strategy_to_json_dict():
    backend_a = MagicMock(spec=Backend)
    backend_a.describe.return_value = "BackendA"
    backend_b = MagicMock(spec=Backend)
    backend_b.describe.return_value = "BackendB"

    strategy = FirstWinsStrategy(backends=[backend_a, backend_b])
    result = strategy.to_json_dict()

    assert result["backends"] == ["BackendA", "BackendB"]
    _assert_strategy_profiling_defaults(result)


def test_one_backend_strategy_to_json_dict():
    backend = MagicMock(spec=Backend)
    backend.describe.return_value = "SingleBackend"

    strategy = OneBackendStrategy(backend=backend)
    result = strategy.to_json_dict()

    assert result["backend"] == "SingleBackend"
    _assert_strategy_profiling_defaults(result)


def test_max_throughput_strategy_to_json_dict():
    backend_a = MagicMock(spec=Backend)
    backend_a.describe.return_value = "BackendA"
    backend_b = MagicMock(spec=Backend)
    backend_b.describe.return_value = "BackendB"

    strategy = MaxThroughputStrategy(backends=[backend_a, backend_b])
    strategy.enable_performance_validation(False)
    result = strategy.to_json_dict()

    assert result["backends"] == ["BackendA", "BackendB"]
    _assert_strategy_profiling_defaults(result)
