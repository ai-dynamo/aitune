# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from unittest.mock import Mock

import pytest
import torch

from aitune.torch.backend import Backend
from aitune.torch.backend.torch_eager import TorchEagerBackend
from aitune.torch.distributed import DistributedOutcome
from aitune.torch.task.find_max_batch_size import find_max_throughput_for_backend
from aitune.torch.task.profiling import NumStepsMeasuringStopStrategy, ProfilingConfig, ProfilingResultEvent
from aitune.torch.task.profiling.measuring_strategy import MeasuringStrategy
from aitune.torch.task.profiling.metrics import BoundaryLatencies
from aitune.torch.task.profiling.profiling import ProfilingStatus, profile, profile_backend
from aitune.torch.task.profiling.profiling_stop_strategy import (
    AllSamplesProfilingStopStrategy,
    ThroughputSaturatedProfilingStopStrategy,
)
from aitune.torch.tune_strategy.max_throughput_strategy import MaxThroughputProfilingResult, MaxThroughputStrategy
from aitune.torch.tune_strategy.min_latency_strategy import MinLatencyStrategy
from aitune.torch.utils.shape_samples import shape_boundary_samples
from tests.toy_backends import SleepBackend
from tests.utilities.helpers import make_graph_spec, make_sample_store, update_input_spec


class ShapeTimings(MeasuringStrategy):
    def __init__(self, latencies, warmup=False):
        self.latencies = latencies
        self.warmup = warmup
        self.calls = {}
        self.order = []

    def do_measurement(self, batch_size, model, sample, **kwargs):
        args, call_kwargs = sample
        model(*args, **call_kwargs)
        width = args[0].shape[1]
        assert args[0].shape[0] == batch_size
        key = (batch_size, width)
        self.calls[key] = self.calls.get(key, 0) + 1
        if not self.order or self.order[-1] != key:
            self.order.append(key)
        latency = self.latencies[batch_size][width // 4 - 1]
        if latency is None:
            raise RuntimeError("Cannot measure this shape")
        if self.warmup and self.calls[key] == 1:
            latency = 1000
        return [ProfilingResultEvent(0, "model", "backend", batch_size, "inference", execution_time=latency * 1e6)]


def setup_profile(tmp_path, latencies, *, static=False, stop=None, warmup=False):
    model = torch.nn.Identity()
    first = ((torch.ones(1, 4),), {})
    graph = make_graph_spec(model.forward, first, batch_size=1)
    width = 4 if static else 8
    update_input_spec(graph, ((torch.ones(1, width),), {}), batch_size=1)
    update_input_spec(graph, ((torch.ones(4, width),), {}), batch_size=4)
    timings = ShapeTimings(latencies, warmup=warmup)
    config = ProfilingConfig(
        batch_sizes=list(latencies),
        measuring_strategy=timings,
        measurement_stop_strategy=NumStepsMeasuringStopStrategy(num_steps=1, warmup_samples=1),
        profiling_stop_strategy=stop or AllSamplesProfilingStopStrategy(),
    )
    backend = Mock(spec=Backend)
    backend.describe.return_value = "backend"
    return backend, graph, make_sample_store([first], tmp_path), config, timings


def test_both_shapes_are_measured_within_each_batch_and_warmups_are_separate(tmp_path):
    backend, graph, samples, config, timings = setup_profile(tmp_path, {1: (2, 8), 2: (4, 18)}, warmup=True)
    status = profile_backend(
        backend,
        "model",
        graph,
        samples,
        config,
        sample_preparer=partial(shape_boundary_samples, graph_spec=graph, preserve_batch_axes=config.batching),
    )

    assert status.status is ProfilingStatus.Status.SUCCESS
    assert timings.order == [(1, 4), (1, 8), (2, 4), (2, 8)]
    assert status.results.batch_latencies(config.measurement_stop_strategy) == {
        1: (2, 8),
        2: (4, 18),
    }


def test_static_shapes_are_measured_once(tmp_path):
    backend, graph, samples, config, timings = setup_profile(tmp_path, {1: (2, 2), 2: (4, 4)}, static=True)
    status = profile_backend(
        backend,
        "model",
        graph,
        samples,
        config,
        sample_preparer=partial(shape_boundary_samples, graph_spec=graph, preserve_batch_axes=config.batching),
    )

    assert timings.order == [(1, 4), (2, 4)]
    assert all(len(batch.sample_events) == 1 for batch in status.results.batches)
    assert status.results.batch_latencies(config.measurement_stop_strategy)[1] == (2,)


def test_empty_recorded_axis_expands_to_max_shape(torch_device):
    model = torch.nn.Identity()
    first = ((torch.empty(1, 0, dtype=torch.int64, device=torch_device),), {})
    graph = make_graph_spec(model.forward, first, batch_size=1)
    larger = torch.ones(1, 8, dtype=torch.int64, device=torch_device)
    update_input_spec(graph, ((larger,), {}), batch_size=1)

    (min_args, _), (max_args, _) = shape_boundary_samples(first, graph, preserve_batch_axes=True)

    assert min_args[0].shape == (1, 0)
    torch.testing.assert_close(max_args[0], torch.zeros_like(larger))


def test_backend_profile_defaults_to_one_recorded_sample_even_with_dynamic_shapes(tmp_path):
    backend, graph, samples, config, timings = setup_profile(tmp_path, {1: (2, 8), 2: (4, 18)})

    status = profile_backend(backend, "model", graph, samples, config)

    assert status.status is ProfilingStatus.Status.SUCCESS
    assert timings.order == [(1, 4), (2, 4)]
    assert len(status.results.entries) == 4
    assert status.results.batch_latencies(config.measurement_stop_strategy) == {1: (2,), 2: (4,)}


def test_backend_profile_can_measure_three_prepared_samples_per_batch(tmp_path):
    backend, graph, samples, config, timings = setup_profile(tmp_path, {1: (2, 8, 18), 2: (4, 18, 32)}, warmup=True)

    def prepare_samples(sample):
        return tuple(((torch.ones(1, width),), {}) for width in (4, 8, 12))

    status = profile_backend(backend, "model", graph, samples, config, sample_preparer=prepare_samples)

    assert status.status is ProfilingStatus.Status.SUCCESS
    assert timings.order == [(batch, width) for batch in (1, 2) for width in (4, 8, 12)]
    assert status.results.batch_latencies(config.measurement_stop_strategy) == {
        1: (2, 8, 18),
        2: (4, 18, 32),
    }
    assert len(status.results.entries) == 12


def test_prepared_sample_group_must_not_be_empty(tmp_path):
    backend, graph, samples, config, timings = setup_profile(tmp_path, {1: (2, 8)})

    status = profile_backend(backend, "model", graph, samples, config, sample_preparer=lambda sample: ())

    assert status.status is ProfilingStatus.Status.FAILURE
    assert "at least one sample" in str(status.error)
    assert timings.order == []


def test_failed_max_shape_keeps_only_earlier_complete_batch_pairs(tmp_path):
    backend, graph, samples, config, _ = setup_profile(tmp_path, {1: (2, 8), 2: (4, None)})

    status = profile_backend(
        backend,
        "model",
        graph,
        samples,
        config,
        sample_preparer=partial(shape_boundary_samples, graph_spec=graph, preserve_batch_axes=config.batching),
    )

    assert [batch.batch_size for batch in status.results.batches] == [1]
    batch = status.results.batches[0]
    assert len(batch.sample_events) == 2
    assert all(event.batch_size == 1 for event in batch.sample_events[0] + batch.sample_events[1])


def test_generic_dataset_profile_measures_one_sample_per_batch():
    timings = ShapeTimings({1: (2, 2), 2: (4, 4)})
    config = ProfilingConfig(
        batch_sizes=[1, 2],
        measuring_strategy=timings,
        measurement_stop_strategy=NumStepsMeasuringStopStrategy(num_steps=1, warmup_samples=1),
        profiling_stop_strategy=AllSamplesProfilingStopStrategy(),
    )

    status = profile(Mock(), torch.ones(4, 4), config)

    assert status.status is ProfilingStatus.Status.SUCCESS
    assert timings.order == [(1, 4), (2, 4)]
    assert all(len(batch.sample_events) == 1 for batch in status.results.batches)
    assert [event.batch_size for event in status.results.entries] == [1, 1, 2, 2]


@pytest.mark.parametrize(
    "latencies,expected_batches",
    [
        ({1: (10, 10), 2: (20, 10), 4: (1, 1)}, (1, 2)),
        ({1: (10, 10), 2: (10, 20), 4: (20, 30)}, (1, 2, 4)),
    ],
)
def test_saturation_uses_min_shape_after_collecting_both(tmp_path, latencies, expected_batches):
    backend, graph, samples, config, timings = setup_profile(
        tmp_path,
        latencies,
        stop=ThroughputSaturatedProfilingStopStrategy(throughput_backoff_limit=0),
    )
    status = profile_backend(
        backend,
        "model",
        graph,
        samples,
        config,
        sample_preparer=partial(shape_boundary_samples, graph_spec=graph, preserve_batch_axes=config.batching),
    )

    assert status.status is ProfilingStatus.Status.SUCCESS
    assert timings.order == [(batch, width) for batch in expected_batches for width in (4, 8)]


def test_combined_score_selects_batch_but_reports_min_shape(tmp_path):
    backend, graph, samples, config, _ = setup_profile(tmp_path, {1: (2, 18), 2: (6, 8)})
    strategy = MaxThroughputStrategy(backends=[backend], profiling_config=config)
    result = strategy._measure(backend, "model", graph, samples, config)

    assert result.selected_batch_size == 2
    assert result.metric == pytest.approx(2000 / 6)
    assert result.latency == pytest.approx(6)
    assert result.selection_score == pytest.approx(2000 / (6 * 8) ** 0.5)
    assert result.to_json_dict("throughput") == {
        "throughput": pytest.approx(2000 / 6),
        "latency": pytest.approx(6),
        "selected_batch_size": 2,
    }


def test_max_throughput_selects_batch_using_worst_rank_at_each_boundary(tmp_path, mocker):
    _, graph, samples, config, _ = setup_profile(tmp_path, {1: (1, 100), 2: (60, 60)})
    peer_latencies = {1: BoundaryLatencies(25, 25), 2: BoundaryLatencies(60, 60)}

    def collect_results(local_latencies, local_error):
        assert local_error is None
        return DistributedOutcome((None, None)), [local_latencies, peer_latencies]

    coordinator = mocker.patch("aitune.torch.tune_strategy.max_throughput_strategy.coordinator")
    coordinator.collect_results.side_effect = collect_results
    strategy = MaxThroughputStrategy(backends=[TorchEagerBackend()], profiling_config=config)
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    strategy.enable_performance_validation(False)

    strategy.tune(torch.nn.Identity(), "model", graph, samples, torch.device("cpu"), tmp_path)

    measurements = strategy.backend_results
    assert measurements is not None
    assert measurements[0]["selected_batch_size"] == 2
    assert measurements[0]["throughput"] == pytest.approx(2000 / 60)
    assert measurements[0]["latency"] == pytest.approx(60)


def test_batch_discovery_keeps_min_shape_policy_while_strategy_uses_both_shapes(tmp_path):
    backend, graph, samples, config, _ = setup_profile(tmp_path, {1: (2, 18), 2: (6, 8)})

    discovery_batch, throughput, _ = find_max_throughput_for_backend(backend, "model", graph, samples, config)
    strategy = MaxThroughputStrategy(backends=[backend], profiling_config=config)
    result = strategy._measure(backend, "model", graph, samples, config)

    assert discovery_batch == 1
    assert throughput == pytest.approx(500)
    assert result.selected_batch_size == 2
    assert result.throughput == pytest.approx(2000 / 6)


def test_min_latency_score_and_report_are_separate(tmp_path):
    backend, graph, samples, config, _ = setup_profile(tmp_path, {1: (2, 18)})
    strategy = MinLatencyStrategy(backends=[backend], profiling_config=config)
    result = strategy._measure(backend, "model", graph, samples, config)

    assert result.metric == pytest.approx(2)
    assert result.selection_score == pytest.approx(6)
    assert strategy._fmt_result(result) == "2.000 ms"


@pytest.mark.parametrize(
    "strategy_type,result_type",
    [
        (MaxThroughputStrategy, MaxThroughputProfilingResult),
    ],
)
def test_throughput_boundary_validation_uses_each_batch_size_regardless_of_report_label(
    monkeypatch, strategy_type, result_type
):
    backend = Mock(spec=Backend)
    strategy = strategy_type(backends=[backend])
    monkeypatch.setattr(strategy, "_metric_label", "latency")
    strategy._baseline_result = result_type(
        selected_batch_size=1, throughput=100, latency=10, boundary_latencies=BoundaryLatencies(10, 10)
    )
    result = result_type(
        selected_batch_size=2, throughput=200, latency=10, boundary_latencies=BoundaryLatencies(10, 20)
    )

    comparison = strategy._record_perf_result(backend, result)

    assert comparison is not None
    assert comparison.speedup == pytest.approx(2)
    assert comparison.passed


@pytest.mark.parametrize("mode", ["enabled", "diagnostic"])
def test_measurement_keeps_regressing_result_for_baseline_comparison(tmp_path, mode):
    backend, graph, samples, config, _ = setup_profile(tmp_path, {1: (1, 12), 2: (10, 10)})
    strategy = MaxThroughputStrategy(backends=[backend], profiling_config=config)
    strategy.enable_performance_validation(mode)
    strategy._baseline_result = MaxThroughputProfilingResult(
        selected_batch_size=1,
        throughput=100,
        latency=10,
        boundary_latencies=BoundaryLatencies(10, 10),
    )
    result = strategy._measure(backend, "model", graph, samples, config)

    assert result.selected_batch_size == 1
    assert result.metric == pytest.approx(1000)
    comparison = strategy._record_perf_result(backend, result)
    assert comparison is not None
    assert comparison.passed is False
    assert comparison.speedup == pytest.approx(10)
    assert strategy.perf_validation_results == [comparison]


@pytest.mark.parametrize("mode", ["enabled", "diagnostic"])
def test_backend_comparison_records_results_before_excluding_candidate(tmp_path, monkeypatch, mode):
    backend, graph, samples, config, _ = setup_profile(tmp_path, {1: (1, 12), 2: (10, 10)})
    backend.is_active = False
    strategy = MaxThroughputStrategy(backends=[backend], profiling_config=config)
    strategy.enable_performance_validation(mode)
    strategy._baseline_result = MaxThroughputProfilingResult(
        selected_batch_size=1,
        throughput=100,
        latency=10,
        boundary_latencies=BoundaryLatencies(10, 10),
    )
    strategy.backend_results = [{"backend": "backend", "success": True}]
    monkeypatch.setattr(strategy, "_build_and_validate_backend", lambda *args: backend)

    selected = strategy._run_backends(
        torch.nn.Identity(), "model", graph, samples, torch.device("cpu"), tmp_path, True, 4
    )

    assert strategy.backend_results[0]["throughput"] == pytest.approx(1000)
    assert strategy.backend_results[0]["selected_batch_size"] == 1
    assert len(strategy.perf_validation_results) == 1
    assert strategy.perf_validation_results[0].passed is False
    assert strategy.perf_validation_results[0].speedup == pytest.approx(10)
    if mode == "enabled":
        assert selected is None
    else:
        assert selected is not None
        assert selected.backend is backend


def test_backend_selection_uses_score_while_speedup_logging_uses_min_shape(tmp_path):
    backend, graph, samples, config, _ = setup_profile(tmp_path, {1: (2, 18), 2: (6, 8)})
    strategy = MaxThroughputStrategy(backends=[backend], profiling_config=config)
    result = strategy._measure(backend, "model", graph, samples, config)
    baseline = MaxThroughputProfilingResult(
        selected_batch_size=1,
        throughput=350,
        latency=1000 / 350,
        boundary_latencies=BoundaryLatencies(1000 / 350, 18),
    )
    strategy._baseline_result = baseline

    assert strategy._is_better(result, baseline)
    assert result.boundary_latencies is not None
    assert strategy._passes_boundary_validation(result.selected_batch_size, result.boundary_latencies)
    comparison = strategy._record_perf_result(backend, result)
    assert comparison is not None
    assert comparison.speedup == pytest.approx((2000 / 6) / 350)
    assert "333.33 samples/s" in strategy._fmt_result(result)
    assert "geometric" not in strategy._fmt_result(result)


@pytest.mark.parametrize(
    "latencies,number,label,passed",
    [((5, 20), "2.00x", "faster", False), ((10.4, 5), "0.96x", "slower", True)],
)
def test_min_latency_log_labels_min_shape_speedup(tmp_path, latencies, number, label, passed):
    _, graph, samples, config, candidate_timings = setup_profile(tmp_path, {1: latencies})
    eager_timings = ShapeTimings({1: (10, 10)})

    class BackendTimings(MeasuringStrategy):
        def do_measurement(self, batch_size, model, sample, **kwargs):
            timings = eager_timings if kwargs["backend_details"].startswith("TorchEagerBackend(") else candidate_timings
            return timings.do_measurement(batch_size, model, sample, **kwargs)

    messages = []
    config.measuring_strategy = BackendTimings()
    strategy = MinLatencyStrategy(
        backends=[SleepBackend(sleep_time=0)],
        profiling_config=config,
        sink=lambda message, *args: messages.append(message % args if args else message),
    )
    strategy.enable_correctness_check(False)
    selected = strategy.tune(torch.nn.Identity(), "model", graph, samples, torch.device("cpu"), tmp_path)

    message = next(message for message in messages if "backend profiled" in message)
    assert f"speedup={number}" in message
    assert label in message
    assert strategy.perf_validation_results[0].passed is passed
    assert isinstance(selected, TorchEagerBackend if not passed else SleepBackend)
