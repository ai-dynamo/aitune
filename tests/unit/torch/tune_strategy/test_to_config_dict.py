# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TuneStrategy.to_json_dict and concrete overrides."""

from unittest.mock import MagicMock

from aitune.torch.backend.backend import Backend
from aitune.torch.tune_strategy.first_wins_strategy import FirstWinsStrategy
from aitune.torch.tune_strategy.max_throughput_strategy import MaxThroughputStrategy
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy
from aitune.torch.tune_strategy.tune_strategy import DummyTuneStrategy


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

    assert result == {"backends": ["BackendA", "BackendB"]}


def test_one_backend_strategy_to_json_dict():
    backend = MagicMock(spec=Backend)
    backend.describe.return_value = "SingleBackend"

    strategy = OneBackendStrategy(backend=backend)
    result = strategy.to_json_dict()

    assert result == {"backend": "SingleBackend"}


def test_max_throughput_strategy_to_json_dict():
    backend_a = MagicMock(spec=Backend)
    backend_a.describe.return_value = "BackendA"
    backend_b = MagicMock(spec=Backend)
    backend_b.describe.return_value = "BackendB"

    strategy = MaxThroughputStrategy(backends=[backend_a, backend_b])
    strategy.enable_validate_against_baseline(False)
    result = strategy.to_json_dict()

    assert result == {
        "backends": ["BackendA", "BackendB"],
        "measurement_stop_strategy": "NumStepsMeasuringStopStrategy",
        "profiling_stop_strategy": "ThroughputSaturatedProfilingStopStrategy",
    }
