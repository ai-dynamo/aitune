# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from unittest.mock import Mock

import pytest
import torch

from aitune.torch.backend import Backend
from aitune.torch.backend.torch_eager import TorchEagerBackend
from aitune.torch.task.profiling import NumStepsMeasuringStopStrategy, ProfilingConfig, ProfilingResultEvent
from aitune.torch.task.profiling.measuring_strategy import MeasuringStrategy
from aitune.torch.task.profiling.profiling import ProfilingStatus, profile, profile_backend
from aitune.torch.task.profiling.profiling_stop_strategy import (
    AllSamplesProfilingStopStrategy,
    ThroughputSaturatedProfilingStopStrategy,
)
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


def test_min_latency_score_and_report_are_separate(tmp_path):
    backend, graph, samples, config, _ = setup_profile(tmp_path, {1: (2, 18)})
    strategy = MinLatencyStrategy(backends=[backend], profiling_config=config)
    result = strategy._measure(backend, "model", graph, samples, config)

    assert result.metric == pytest.approx(2)
    assert result.selection_score == pytest.approx(6)
    assert strategy._fmt_result(result) == "2.000 ms"


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
