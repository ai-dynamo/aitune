# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from aitune.torch.backend import Backend
from aitune.torch.backend.torch_eager import TorchEagerBackend
from aitune.torch.task.profiling import (
    ModelExecutionTimeMeasuringStrategy,
    NumStepsMeasuringStopStrategy,
    ProfilingConfig,
)
from aitune.torch.task.profiling.profiling_stop_strategy import AllSamplesProfilingStopStrategy
from aitune.torch.tune_strategy.profiling_tune_strategy import (
    BackendPerfResult,
    BackendProfilingResult,
    ProfilingTuneStrategy,
    _TuneCandidate,
)
from tests.toy_backends import SleepBackend
from tests.toy_models.torch_models import ToyTorchModel


def _profiling_config() -> ProfilingConfig:
    return ProfilingConfig(
        batch_sizes=[1],
        measuring_strategy=ModelExecutionTimeMeasuringStrategy(),
        measurement_stop_strategy=NumStepsMeasuringStopStrategy(num_steps=3),
        profiling_stop_strategy=AllSamplesProfilingStopStrategy(),
    )


class _ControlledStrategy(ProfilingTuneStrategy):
    """Minimal concrete strategy for testing ProfilingTuneStrategy base behaviour.

    _measure always returns a result with self.measure_value so tests can control the measured metric.
    """

    _title = "Controlled"
    _description = "test strategy"
    _metric_label = "metric"
    _metric_unit = "u"
    _value_fmt = ".2f"

    def __init__(self, *args, measure_value: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.measure_value = measure_value
        self.enable_find_max_batch_size(False)

    def _measure(self, backend, name, graph_spec, data, profiling_cfg):
        return _ControlledProfilingResult(metric_value=self.measure_value)

    def _is_better(self, result: BackendProfilingResult, other: BackendProfilingResult) -> bool:
        return result.metric > other.metric

    def _speedup(self, result: BackendProfilingResult, baseline_result: BackendProfilingResult) -> float:
        return result.metric / baseline_result.metric


@dataclass(kw_only=True)
class _ControlledProfilingResult(BackendProfilingResult):
    metric_value: float
    selected_batch_size: int = 1

    @property
    def metric(self) -> float:
        return self.metric_value


@pytest.fixture
def strategy():
    return _ControlledStrategy(backends=[SleepBackend()], profiling_config=_profiling_config())


# ---------------------------------------------------------------------------
# _resolve_winner
# ---------------------------------------------------------------------------


def test_resolve_winner_returns_best_when_faster_than_baseline(strategy):
    """Best candidate beats the baseline → best is returned."""
    baseline = MagicMock(spec=Backend)
    best = _TuneCandidate(backend=MagicMock(spec=Backend), result=_ControlledProfilingResult(metric_value=2.0))
    strategy._baseline_result = _ControlledProfilingResult(metric_value=1.0)
    strategy._baseline_backend = baseline

    result = strategy._resolve_winner(best)

    assert result is best


def test_resolve_winner_falls_back_to_baseline_when_best_is_slower(strategy):
    """Baseline beats the best candidate → baseline is returned."""
    baseline = MagicMock(spec=Backend)
    best = _TuneCandidate(backend=MagicMock(spec=Backend), result=_ControlledProfilingResult(metric_value=0.5))
    strategy._baseline_result = _ControlledProfilingResult(metric_value=1.0)
    strategy._baseline_backend = baseline

    result = strategy._resolve_winner(best)

    assert result.backend is baseline
    assert result.result.metric == 1.0


def test_resolve_winner_falls_back_to_baseline_when_best_ties(strategy):
    """Best candidate must beat the baseline; ties fall back to baseline."""
    baseline = MagicMock(spec=Backend)
    best = _TuneCandidate(backend=MagicMock(spec=Backend), result=_ControlledProfilingResult(metric_value=1.0))
    strategy._baseline_result = _ControlledProfilingResult(metric_value=1.0)
    strategy._baseline_backend = baseline

    result = strategy._resolve_winner(best)

    assert result.backend is baseline
    assert result.result.metric == 1.0


def test_resolve_winner_returns_baseline_when_no_backends_succeed(strategy):
    """All user backends failed (best=None) but baseline exists → baseline is returned."""
    baseline = MagicMock(spec=Backend)
    strategy._baseline_result = _ControlledProfilingResult(metric_value=1.0)
    strategy._baseline_backend = baseline

    result = strategy._resolve_winner(None)

    assert result.backend is baseline


def test_resolve_winner_raises_when_no_backends_and_validation_disabled(strategy):
    """No backends succeeded and validation is disabled → RuntimeError."""
    strategy._performance_validation_enabled = False
    strategy._baseline_result = None
    strategy._baseline_backend = None

    with pytest.raises(RuntimeError, match="No correct backend found"):
        strategy._resolve_winner(None)


def test_resolve_winner_raises_when_best_is_none_and_baseline_failed(strategy):
    """All user backends failed and baseline profiling also failed → RuntimeError."""
    strategy._baseline_result = None
    strategy._baseline_backend = None

    with pytest.raises(RuntimeError, match="No correct backend found"):
        strategy._resolve_winner(None)


# ---------------------------------------------------------------------------
# _record_perf_result
# ---------------------------------------------------------------------------


def test_record_perf_result_skipped_when_baseline_is_none(strategy):
    strategy._baseline_result = None
    strategy._record_perf_result(MagicMock(spec=Backend), _ControlledProfilingResult(metric_value=1.0))
    assert strategy.perf_validation_results == []


def test_record_perf_result_skipped_when_baseline_is_zero(strategy):
    strategy._baseline_result = _ControlledProfilingResult(metric_value=0.0)
    strategy._record_perf_result(MagicMock(spec=Backend), _ControlledProfilingResult(metric_value=1.0))
    assert strategy.perf_validation_results == []


def test_record_perf_result_skipped_when_value_is_zero(strategy):
    strategy._baseline_result = _ControlledProfilingResult(metric_value=1.0)
    strategy._record_perf_result(MagicMock(spec=Backend), _ControlledProfilingResult(metric_value=0.0))
    assert strategy.perf_validation_results == []


def test_record_perf_result_populates_result_correctly(strategy):
    backend = MagicMock(spec=Backend)
    backend.describe.return_value = "mock"
    strategy._baseline_result = _ControlledProfilingResult(metric_value=1.0)

    strategy._record_perf_result(backend, _ControlledProfilingResult(metric_value=2.0))

    assert len(strategy.perf_validation_results) == 1
    result = strategy.perf_validation_results[0]
    assert result.backend_description == "mock"
    assert result.metric == pytest.approx(2.0)
    assert result.baseline_metric == pytest.approx(1.0)
    assert result.speedup == pytest.approx(2.0)
    assert result.passed is True


def test_record_perf_result_passed_false_when_slower_than_baseline(strategy):
    backend = MagicMock(spec=Backend)
    backend.describe.return_value = "mock"
    strategy._baseline_result = _ControlledProfilingResult(metric_value=2.0)

    strategy._record_perf_result(backend, _ControlledProfilingResult(metric_value=1.0))

    assert strategy.perf_validation_results[0].passed is False
    assert strategy.perf_validation_results[0].speedup == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _pre_tune state management
# ---------------------------------------------------------------------------


def test_pre_tune_profiles_baseline_when_validation_enabled(torch_device, tmp_path):
    """_pre_tune sets _baseline_result and _baseline_backend when validation is enabled."""
    strategy = _ControlledStrategy(backends=[SleepBackend()], profiling_config=_profiling_config(), measure_value=5.0)
    strategy.enable_correctness_check(False)
    model = ToyTorchModel()
    graph_spec = model.graph_spec(batch_sizes=[1], device=torch_device)
    data = [((model.sample().unsqueeze(0),), {})]

    strategy._pre_tune(model, "test", graph_spec, data, torch_device, tmp_path)

    assert strategy._baseline_result.metric == pytest.approx(5.0)
    assert isinstance(strategy._baseline_backend, TorchEagerBackend)


def test_pre_tune_skips_baseline_when_validation_disabled(torch_device, tmp_path):
    """_pre_tune leaves _baseline_result as None when performance validation is disabled."""
    strategy = _ControlledStrategy(backends=[SleepBackend()], profiling_config=_profiling_config())
    strategy.enable_performance_validation(False)
    strategy.enable_correctness_check(False)
    model = ToyTorchModel()
    graph_spec = model.graph_spec(batch_sizes=[1], device=torch_device)
    data = [((model.sample().unsqueeze(0),), {})]

    strategy._pre_tune(model, "test", graph_spec, data, torch_device, tmp_path)

    assert strategy._baseline_result is None
    assert strategy._baseline_backend is None


def test_pre_tune_resets_state_on_each_call(torch_device, tmp_path):
    """perf_validation_results and baseline fields are reset at the start of each _pre_tune."""
    strategy = _ControlledStrategy(backends=[SleepBackend()], profiling_config=_profiling_config())
    strategy.enable_correctness_check(False)

    # Inject stale state from a previous run
    strategy.perf_validation_results = [BackendPerfResult("stale", 1.0, 1.0, 1.0, True)]
    strategy._baseline_result = _ControlledProfilingResult(metric_value=99.0)

    model = ToyTorchModel()
    graph_spec = model.graph_spec(batch_sizes=[1], device=torch_device)
    data = [((model.sample().unsqueeze(0),), {})]

    strategy._pre_tune(model, "test", graph_spec, data, torch_device, tmp_path)

    assert strategy.perf_validation_results == []
    assert strategy._baseline_result.metric != 99.0


def test_init_preserves_explicit_empty_backend_list():
    """An explicit empty backend list means no user backends, not default backends."""
    strategy = _ControlledStrategy(backends=[], profiling_config=_profiling_config())

    assert strategy._backends == []


# ---------------------------------------------------------------------------
# _fmt
# ---------------------------------------------------------------------------


def test_fmt_formats_value_with_unit(strategy):
    assert strategy._fmt(12.3456) == "12.35 u"


def test_fmt_uses_subclass_unit_and_format():
    from aitune.torch.tune_strategy.max_throughput_strategy import MaxThroughputStrategy
    from aitune.torch.tune_strategy.min_latency_strategy import MinLatencyStrategy

    dummy = [SleepBackend()]
    assert MaxThroughputStrategy(backends=dummy)._fmt(100.5) == "100.50 samples/s"
    assert MinLatencyStrategy(backends=dummy)._fmt(5.1234) == "5.123 ms"
