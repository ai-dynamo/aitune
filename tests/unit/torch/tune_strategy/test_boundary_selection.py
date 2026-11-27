# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import pytest
import torch

from aitune.exceptions import AITuneUserInputError
from aitune.torch.backend import TorchEagerBackend
from aitune.torch.task.profiling import ProfilingConfig
from aitune.torch.task.profiling.events import ProfilingResultEvent
from aitune.torch.task.profiling.measuring_stop_strategy import StableWindowMeasuringStopStrategy
from aitune.torch.task.profiling.measuring_strategy import MeasuringStrategy
from aitune.torch.tune_strategy.latency_budget_strategy import LatencyBudgetStrategy
from aitune.torch.tune_strategy.max_throughput_strategy import MaxThroughputStrategy
from aitune.torch.tune_strategy.min_latency_strategy import MinLatencyStrategy
from tests.toy_backends import SleepBackend
from tests.utilities.helpers import make_graph_spec, make_sample_store, update_input_spec


@pytest.mark.parametrize(
    "strategy_type", [MinLatencyStrategy, MaxThroughputStrategy, partial(LatencyBudgetStrategy, latency_budget_ms=50)]
)
@pytest.mark.parametrize("sizes", [(8, 16), (16, 8)])
def test_shape_crossover_rejects_regression_before_ranking(tmp_path, strategy_type, sizes):
    strategy, selected = _tune(tmp_path, strategy_type, sizes, ["crossover", "safe"])

    assert isinstance(selected, _ShapeBackend)
    assert selected.label == "safe"
    crossover, safe = strategy.perf_validation_results
    assert crossover.passed is False
    assert safe.passed is True
    assert crossover.boundary_speedups == pytest.approx({"min_shape": 2.0, "max_shape": 10.0 / 12.0})
    assert safe.speedup == pytest.approx(1.25)


@pytest.mark.parametrize(
    "strategy_type", [MinLatencyStrategy, MaxThroughputStrategy, partial(LatencyBudgetStrategy, latency_budget_ms=50)]
)
def test_shape_crossover_falls_back_to_eager_when_no_candidate_qualifies(tmp_path, strategy_type):
    _, selected = _tune(tmp_path, strategy_type, (8, 16), ["crossover"])
    assert isinstance(selected, TorchEagerBackend)


@pytest.mark.parametrize("mode", ["diagnostic", "disabled"])
def test_non_gating_modes_preserve_objective_selection(tmp_path, mode):
    strategy, selected = _tune(tmp_path, MinLatencyStrategy, (8, 16), ["crossover", "safe"], mode=mode)
    assert isinstance(selected, _ShapeBackend)
    assert selected.label == "crossover"
    if mode == "diagnostic":
        assert strategy.perf_validation_results[0].passed is False
    else:
        assert not strategy.perf_validation_results


def test_failed_boundary_measurement_keeps_other_candidates_eligible(tmp_path):
    strategy, selected = _tune(tmp_path, MinLatencyStrategy, (8, 16), ["broken", "safe"])
    assert isinstance(selected, _ShapeBackend)
    assert selected.label == "safe"
    assert strategy.backend_results[0]["profiling_error"]


@pytest.mark.parametrize("budget,label", [(9.0, "crossover"), (10.1, "tolerated")])
def test_latency_budget_rejects_candidate_that_exceeds_budget_only_at_max_shape(tmp_path, budget, label):
    strategy, selected = _tune(
        tmp_path, partial(LatencyBudgetStrategy, latency_budget_ms=budget), (8, 16), [label, "safe"]
    )

    assert isinstance(selected, _ShapeBackend)
    assert selected.label == "safe"
    assert "max_shape" in strategy.backend_results[0]["profiling_error"]
    assert strategy.backend_results[1]["shape_boundaries"]["max_shape"]["latency"] <= budget


@pytest.mark.parametrize(
    "strategy_type", [MinLatencyStrategy, MaxThroughputStrategy, partial(LatencyBudgetStrategy, latency_budget_ms=50)]
)
@pytest.mark.parametrize("sizes", [(8, 16), (16, 8), (12, 8, 16)])
def test_geometric_mean_wins_over_first_shape_score(tmp_path, strategy_type, sizes):
    _, selected = _tune(tmp_path, strategy_type, sizes, ["min_only", "safe"])
    assert isinstance(selected, _ShapeBackend)
    assert selected.label == "safe"


@pytest.mark.parametrize(
    "strategy_type", [MinLatencyStrategy, MaxThroughputStrategy, partial(LatencyBudgetStrategy, latency_budget_ms=50)]
)
def test_reporting_uses_max_shape_while_selection_keeps_both_points(tmp_path, strategy_type):
    strategy, _ = _tune(tmp_path, strategy_type, (8, 16), ["min_only"])

    reported = strategy.backend_results[0]
    assert reported["latency"] == pytest.approx(10.0)
    assert reported["shape_boundaries"]["min_shape"]["latency"] == pytest.approx(7.0)
    assert reported["shape_boundaries"]["max_shape"]["latency"] == pytest.approx(10.0)
    assert strategy.perf_validation_results[0].boundary_speedups == pytest.approx({
        "min_shape": 10.0 / 7.0,
        "max_shape": 1.0,
    })
    assert strategy.perf_validation_results[0].speedup == pytest.approx((10.0 / 7.0) ** 0.5)


def test_small_boundary_regression_is_allowed_when_overall_score_wins(tmp_path):
    strategy, selected = _tune(tmp_path, MinLatencyStrategy, (8, 16), ["tolerated", "safe"])
    assert isinstance(selected, _ShapeBackend)
    assert selected.label == "tolerated"
    comparison = strategy.perf_validation_results[0]
    assert comparison.passed
    assert 0.95 < comparison.boundary_speedups["max_shape"] < 1.0


def test_boundary_tolerance_is_configurable(tmp_path):
    _, selected = _tune(tmp_path, MinLatencyStrategy, (8, 16), ["tolerated", "safe"], min_boundary_speedup=1.0)
    assert isinstance(selected, _ShapeBackend)
    assert selected.label == "safe"


def test_boundary_at_threshold_is_eligible(tmp_path):
    strategy, selected = _tune(tmp_path, MinLatencyStrategy, (8, 16), ["threshold"])
    assert isinstance(selected, _ShapeBackend)
    assert strategy.perf_validation_results[0].passed
    assert strategy.perf_validation_results[0].boundary_speedups["max_shape"] == pytest.approx(0.95)


@pytest.mark.parametrize("threshold", [0, -1, float("nan"), float("inf")])
def test_invalid_boundary_threshold_is_rejected(threshold):
    with pytest.raises(AITuneUserInputError, match="finite and positive"):
        MinLatencyStrategy(backends=[], min_boundary_speedup=threshold)


def test_static_shape_is_profiled_once(tmp_path):
    strategy, _ = _tune(tmp_path, MinLatencyStrategy, (8,), ["safe"])
    assert strategy.perf_validation_results[0].boundary_speedups == pytest.approx({"min_shape": 1.25})


def test_batch_only_bounds_do_not_repeat_the_same_profile(tmp_path):
    strategy, _ = _tune(tmp_path, MinLatencyStrategy, (8, 8), ["safe"], batch_sizes=(1, 2))
    assert strategy.perf_validation_results[0].boundary_speedups == pytest.approx({"min_shape": 1.25})
    measurements = strategy.profiling_config.measuring_strategy
    assert isinstance(measurements, _ShapeMeasurements)
    assert len(measurements.calls) == 2
    assert set(measurements.calls.values()) == {3}


@pytest.mark.parametrize("label", ["slightly_slower", "tie"])
def test_eager_wins_when_eligible_candidate_does_not_improve_overall_score(tmp_path, label):
    strategy, selected = _tune(tmp_path, MinLatencyStrategy, (8, 16), [label])
    assert isinstance(selected, TorchEagerBackend)
    assert strategy.perf_validation_results[0].passed


def test_each_boundary_uses_one_profile_and_excludes_unstable_warmup(tmp_path):
    strategy, _ = _tune(tmp_path, MinLatencyStrategy, (8, 16), ["safe"])
    measurements = strategy.profiling_config.measuring_strategy
    assert isinstance(measurements, _ShapeMeasurements)
    # One warmup and a stable window of two measurements per backend and shape.
    assert len(measurements.calls) == 4
    assert set(measurements.calls.values()) == {3}
    assert strategy.perf_validation_results[0].boundary_speedups == pytest.approx({
        "min_shape": 1.25,
        "max_shape": 1.25,
    })


def test_selection_uses_final_mean_when_profiler_extends_measurement(tmp_path):
    strategy, _ = _tune(tmp_path, MinLatencyStrategy, (8, 16), ["settling"])
    measurements = strategy.profiling_config.measuring_strategy
    assert isinstance(measurements, _ShapeMeasurements)
    assert measurements.calls[("settling", 8)] == 5
    assert measurements.calls[("settling", 16)] == 5
    assert strategy.perf_validation_results[0].speedup == pytest.approx(1.25)


class _ShapeBackend(SleepBackend):
    _build_mode = SleepBackend._build_mode
    _execution_modes = SleepBackend._execution_modes

    def __init__(self, label):
        super().__init__(sleep_time=0)
        self.label = label

    def describe(self):
        return self.label

    def key(self):
        return self.label


class _ShapeMeasurements(MeasuringStrategy):
    def __init__(self):
        self.calls = {}

    def do_measurement(self, batch_size, model, sample, **kwargs):
        args, call_kwargs = sample
        model(*args, **call_kwargs)
        description = kwargs["backend_details"]
        size = args[0].shape[-1]
        latencies = {
            "safe": (8.0, 8.0),
            "settling": (8.0, 8.0),
            "crossover": (5.0, 12.0),
            "min_only": (7.0, 10.0),
            "tolerated": (5.0, 10.2),
            "threshold": (5.0, 10.0 / 0.95),
            "slightly_slower": (10.1, 10.1),
            "broken": (4.0, 4.0),
        }
        if description == "broken" and size == 16:
            raise RuntimeError("Cannot measure the largest shape")
        shape_index = 0 if size == 8 else 1
        latency = latencies.get(description, (10.0, 10.0))[shape_index]
        key = (description, size)
        self.calls[key] = self.calls.get(key, 0) + 1
        if self.calls[key] == 1:
            latency = 1000.0  # Warmup must not contribute to the selected mean.
        elif description == "settling" and self.calls[key] in (2, 3):
            latency *= 0.8 if self.calls[key] == 2 else 1.2
        return [
            ProfilingResultEvent(
                timestamp=0,
                model_name="model",
                backend_details=description,
                batch_size=batch_size,
                phase="inference",
                execution_time=latency * 1e6,
            )
        ]


def _tune(tmp_path, strategy_type, sizes, labels, mode="enabled", min_boundary_speedup=0.95, batch_sizes=None):
    model = torch.nn.Identity()
    batch_sizes = batch_sizes or [1] * len(sizes)
    recorded = []
    for size, batch_size in zip(sizes, batch_sizes, strict=True):
        recorded.append(((torch.zeros(batch_size, 4, size, size),), {}))
    graph_spec = make_graph_spec(model.forward, recorded[0], batch_size=batch_sizes[0])
    for sample, batch_size in zip(recorded[1:], batch_sizes[1:], strict=True):
        update_input_spec(graph_spec, sample, batch_size=batch_size)
    samples = make_sample_store(recorded[:1], tmp_path)
    strategy = strategy_type(
        backends=[_ShapeBackend(label) for label in labels],
        profiling_config=ProfilingConfig(
            batch_sizes=[1],
            measuring_strategy=_ShapeMeasurements(),
            measurement_stop_strategy=StableWindowMeasuringStopStrategy(
                window_size=2, warmup_samples=1, max_cv_ratio=0.01, max_samples=6
            ),
        ),
        min_boundary_speedup=min_boundary_speedup,
    )
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    strategy.enable_performance_validation(mode)
    selected = strategy.tune(model, "spatial-model", graph_spec, samples, torch.device("cpu"), tmp_path)
    return strategy, selected
