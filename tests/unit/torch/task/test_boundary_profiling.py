# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import torch

from aitune.torch.backend import Backend
from aitune.torch.task.profiling import NumStepsMeasuringStopStrategy, ProfilingConfig, ProfilingResultEvent
from aitune.torch.task.profiling.measuring_strategy import MeasuringStrategy
from aitune.torch.task.profiling.profiling import ProfilingStatus, profile, profile_backend
from aitune.torch.task.profiling.profiling_stop_strategy import (
    AllSamplesProfilingStopStrategy,
)
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
