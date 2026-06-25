# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from aitune.torch.backend import Backend
from aitune.torch.backend.torch_eager import TorchEagerBackend
from aitune.torch.task.profiling import (
    ModelExecutionTimeMeasuringStrategy,
    NumStepsMeasuringStopStrategy,
    ProfilingConfig,
    ProfilingResultEvent,
    ProfilingResults,
    ProfilingStatus,
)
from aitune.torch.task.profiling.profiling_stop_strategy import AllSamplesProfilingStopStrategy
from aitune.torch.tune_strategy.latency_budget_strategy import (
    LatencyBudgetProfilingResult,
    LatencyBudgetStrategy,
    _LatencyBudgetProfilingStopStrategy,
)
from tests.toy_backends import SleepBackend
from tests.toy_models.torch_models import ToyTorchModel


@pytest.fixture
def mock_backend():
    """Create a mock backend for testing."""
    backend = MagicMock(spec=Backend)
    backend.name = "mock_backend"
    backend.describe.return_value = "mock_backend"
    return backend


def _profiling_config() -> ProfilingConfig:
    return ProfilingConfig(
        batch_sizes=[1, 2, 4],
        measuring_strategy=ModelExecutionTimeMeasuringStrategy(),
        measurement_stop_strategy=NumStepsMeasuringStopStrategy(num_steps=1, warmup_samples=1),
        profiling_stop_strategy=AllSamplesProfilingStopStrategy(),
    )


def test_describe(mock_backend):
    """Test describe method."""
    strategy = LatencyBudgetStrategy(
        latency_budget_ms=50.0,
        backends=[mock_backend, mock_backend],
        profiling_config=_profiling_config(),
    )
    strategy.enable_performance_validation(False)
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)

    description = strategy.describe()

    assert "name: Latency Budget Strategy" in description
    assert "mock_backend" in description
    assert "latency_budget_ms: 50" in description
    assert description.startswith("name: Latency Budget Strategy")


def test_latency_budget_must_be_positive(mock_backend):
    """Latency budget must be a positive number of milliseconds."""
    with pytest.raises(ValueError, match="positive"):
        LatencyBudgetStrategy(latency_budget_ms=0, backends=[mock_backend])


def test_measure_selects_highest_throughput_under_latency_budget(monkeypatch, mock_backend):
    """_measure filters out batch sizes over budget and returns max remaining throughput."""
    strategy = LatencyBudgetStrategy(
        latency_budget_ms=20.0, backends=[mock_backend], profiling_config=_profiling_config()
    )

    def mock_profile_backend(*args, **kwargs):
        return ProfilingStatus(
            status=ProfilingStatus.Status.SUCCESS,
            results=ProfilingResults(
                entries=[
                    ProfilingResultEvent(0.0, "model", "backend", 1, "inference", execution_time=10e6),
                    ProfilingResultEvent(0.0, "model", "backend", 2, "inference", execution_time=15e6),
                    ProfilingResultEvent(0.0, "model", "backend", 4, "inference", execution_time=25e6),
                ]
            ),
        )

    monkeypatch.setattr("aitune.torch.tune_strategy.latency_budget_strategy.profile_backend", mock_profile_backend)

    result = strategy._measure(mock_backend, "test", MagicMock(), [], _profiling_config())

    assert result.selected_batch_size == 2
    assert result.throughput == pytest.approx(2 / 0.015)
    assert result.latency == pytest.approx(15.0)
    assert result.to_json_dict("throughput") == {
        "throughput": pytest.approx(2 / 0.015),
        "selected_batch_size": 2,
        "latency": pytest.approx(15.0),
    }


def test_measure_raises_when_no_batch_size_satisfies_budget(monkeypatch, mock_backend):
    """_measure raises if every profiled batch size exceeds the latency budget."""
    strategy = LatencyBudgetStrategy(
        latency_budget_ms=5.0, backends=[mock_backend], profiling_config=_profiling_config()
    )

    def mock_profile_backend(*args, **kwargs):
        return ProfilingStatus(
            status=ProfilingStatus.Status.SUCCESS,
            results=ProfilingResults(
                entries=[
                    ProfilingResultEvent(0.0, "model", "backend", 1, "inference", execution_time=10e6),
                    ProfilingResultEvent(0.0, "model", "backend", 2, "inference", execution_time=15e6),
                ]
            ),
        )

    monkeypatch.setattr("aitune.torch.tune_strategy.latency_budget_strategy.profile_backend", mock_profile_backend)

    with pytest.raises(RuntimeError, match="No profile result satisfied latency budget"):
        strategy._measure(mock_backend, "test", MagicMock(), [], _profiling_config())


def test_latency_budget_profiling_stop_strategy_stops_after_budget_is_exceeded():
    """LatencyBudgetStrategy stops profiling larger batch sizes once latency exceeds the budget."""
    stop_strategy = _LatencyBudgetProfilingStopStrategy(
        latency_budget_ms=20.0,
        measurement_stop_strategy=NumStepsMeasuringStopStrategy(num_steps=1, warmup_samples=1),
        base_strategy=AllSamplesProfilingStopStrategy(),
    )

    assert not stop_strategy.should_stop([
        ProfilingResultEvent(0.0, "model", "backend", 1, "inference", execution_time=10e6),
        ProfilingResultEvent(0.0, "model", "backend", 1, "inference", execution_time=10e6),
    ])
    assert stop_strategy.should_stop([
        ProfilingResultEvent(0.0, "model", "backend", 2, "inference", execution_time=10e6),
        ProfilingResultEvent(0.0, "model", "backend", 2, "inference", execution_time=25e6),
    ])


def test_get_profiling_config_wraps_existing_stop_strategy(mock_backend):
    """LatencyBudgetStrategy preserves the configured stop strategy while adding the latency budget stop."""
    profiling_config = _profiling_config()
    strategy = LatencyBudgetStrategy(
        latency_budget_ms=20.0,
        backends=[mock_backend],
        profiling_config=profiling_config,
    )

    result = strategy._get_profiling_config(batching=True, max_batch_size=4)

    assert isinstance(result.profiling_stop_strategy, _LatencyBudgetProfilingStopStrategy)
    assert result.profiling_stop_strategy.base_strategy is profiling_config.profiling_stop_strategy


def test_latency_budget_strategy_selects_max_throughput_backend(torch_device, tmp_path):
    """LatencyBudgetStrategy selects the compliant backend with highest throughput."""
    lower_throughput = SleepBackend(sleep_time=1e-2)
    higher_throughput = SleepBackend(sleep_time=1e-5)
    strategy = LatencyBudgetStrategy(
        latency_budget_ms=50.0,
        backends=[lower_throughput, higher_throughput],
        profiling_config=_profiling_config(),
    )
    strategy.enable_performance_validation(False)
    strategy.enable_correctness_check(False)

    model = ToyTorchModel()
    sample = model.sample().unsqueeze(0)

    def mock_measure(backend, name, graph_spec, data, profiling_cfg):
        return LatencyBudgetProfilingResult(
            selected_batch_size=1,
            throughput=100.0 if backend.sleep_time == lower_throughput.sleep_time else 200.0,
            latency=10.0,
        )

    strategy._measure = mock_measure

    selected = strategy.tune(
        model, "test", model.graph_spec(batch_sizes=[1, 2]), [((sample,), {})], torch_device, tmp_path
    )

    assert isinstance(selected, SleepBackend)
    assert selected.sleep_time == higher_throughput.sleep_time


def test_latency_budget_strategy_raises_when_no_user_backend_satisfies_budget(torch_device, tmp_path):
    """A successful baseline does not hide that no user-provided backend satisfied the budget."""
    strategy = LatencyBudgetStrategy(
        latency_budget_ms=50.0,
        backends=[SleepBackend()],
        profiling_config=_profiling_config(),
    )
    strategy.enable_correctness_check(False)

    model = ToyTorchModel()
    sample = model.sample().unsqueeze(0)

    def mock_measure(backend, name, graph_spec, data, profiling_cfg):
        if isinstance(backend, TorchEagerBackend):
            return LatencyBudgetProfilingResult(selected_batch_size=1, throughput=100.0, latency=10.0)
        raise RuntimeError("budget exceeded")

    strategy._measure = mock_measure

    with pytest.raises(RuntimeError, match="No backend satisfied latency budget"):
        strategy.tune(model, "test", model.graph_spec(batch_sizes=[1, 2]), [((sample,), {})], torch_device, tmp_path)
