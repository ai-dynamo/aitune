# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import pytest

from aitune.torch import Module
from aitune.torch.backend import TorchEagerBackend
from aitune.torch.task.profiling.config import ProfilingConfig
from aitune.torch.task.profiling.events import ProfilingResultEvent
from aitune.torch.task.profiling.measuring_stop_strategy import (
    NumStepsMeasuringStopStrategy,
    StableWindowMeasuringStopStrategy,
)
from aitune.torch.task.profiling.metrics import get_throughput, is_throughput_saturated
from aitune.torch.task.profiling.profiling import ProfilingStatus, profile, profile_backend
from aitune.torch.task.profiling.profiling_stop_strategy import ThroughputSaturatedProfilingStopStrategy
from tests.toy_models.torch_models import ToyTorchModel


def new_event(
    batch_size: int,
    execution_time: float,
    phase: str = "inference",
    backend_details: str = "torch",
    measurement_id: int = 0,
) -> ProfilingResultEvent:
    return ProfilingResultEvent(
        measurement_id=measurement_id,
        timestamp=time.monotonic_ns(),
        batch_size=batch_size,
        execution_time=execution_time,
        phase=phase,
        backend_details=backend_details,
        model_name="test_model:module_name",
    )


def test_get_throughput_batch_size_1():
    profiling_result = [
        new_event(1, 100e6),
        new_event(1, 100e6),
    ]
    assert int(get_throughput(profiling_result)) == 10


def test_get_throughput_batch_size_20():
    profiling_result = [
        new_event(20, 201e6),
        new_event(20, 201e6),
    ]
    assert int(get_throughput(profiling_result)) == 99


def test_is_throughput_saturated_new():
    profiling_result = [
        new_event(1, 100e6),
        new_event(1, 100e6),
    ]
    assert not is_throughput_saturated(profiling_result, 0.01, [])


def test_is_throughput_saturated_next():
    prev_profiling_result = [
        new_event(1, 100e6),
        new_event(1, 100e6),
    ]
    profiling_result = [
        new_event(2, 201e6),
        new_event(2, 201e6),
        new_event(2, 700e6, phase="memory_alloc"),  # memory allocation is not included in throughput calculation
    ]
    assert is_throughput_saturated(profiling_result, 0.01, prev_profiling_result)


def test_throughput_saturated_threshold_strategy():
    strategy = ThroughputSaturatedProfilingStopStrategy(throughput_cutoff_threshold=0.01)
    profiling_result = [
        new_event(1, 100e6),
        new_event(1, 100e6),
    ]
    assert not strategy.should_stop(profiling_result)

    profiling_result = [
        new_event(2, 201e6),
        new_event(2, 201e6),
    ]
    assert strategy.should_stop(profiling_result)


@pytest.mark.parametrize(
    "events_stop,throughput_cutoff_threshold,throughput_backoff_limit",
    [
        (
            [
                ([(1, 150), (1, 150)], False),
                ([(2, 201), (2, 201)], False),
                ([(4, 400), (4, 400)], False),
                ([(8, 796), (8, 801)], False),
                ([(16, 1592), (16, 1601)], True),
            ],
            0.05,
            2,
        ),
        (
            [
                ([(1, 150), (1, 150)], False),  # 150ms/s, first event
                ([(2, 201), (2, 201)], False),  # 100ms/s, no saturation
                ([(4, 400), (4, 400)], False),  # 100ms/s, saturation
                ([(8, 696), (8, 701)], False),  # 87ms/s, no saturation, best
                ([(16, 1492), (16, 1501)], False),  # 93ms/s, saturation, slower
                ([(32, 3084), (32, 3101)], True),  # 96ms/s, again saturation, slower, stop profiling
            ],
            0.05,
            1,
        ),
    ],
)
def test_throughput_saturated_threshold_strategy_with_backoff_and_reset(
    events_stop, throughput_cutoff_threshold, throughput_backoff_limit
):
    strategy = ThroughputSaturatedProfilingStopStrategy(
        throughput_cutoff_threshold=throughput_cutoff_threshold, throughput_backoff_limit=throughput_backoff_limit
    )

    for events, should_stop in events_stop:
        # setup events
        events = [new_event(batch_size, execution_time) for batch_size, execution_time in events]

        # check if should stop
        assert strategy.should_stop(events) == should_stop, f"Events: {events}, should_stop: {should_stop}"

    assert should_stop, "All events should be consumed"


def test_measurement_stop_strategy_num_steps():
    strategy = NumStepsMeasuringStopStrategy(num_steps=10)
    assert not strategy.should_stop([new_event(1, 100e6)])
    for i in range(8):
        assert not strategy.should_stop([new_event(1, 100e6, measurement_id=i)])
    assert strategy._steps_counter == 9
    assert strategy.should_stop([new_event(1, 100e6)])
    assert strategy.should_stop([new_event(1, 100e6)])


def test_measurement_stop_strategy_stable_window():
    strategy = StableWindowMeasuringStopStrategy(window_size=10, stability_percentage=90)
    assert not strategy.should_stop([new_event(1, 100e6)])
    for i in range(8):
        assert not strategy.should_stop([new_event(1, 100e6, measurement_id=i)])
    assert len(strategy._window) == 9
    assert strategy.should_stop([new_event(1, 100e6)])
    assert strategy.should_stop([new_event(1, 100e6)])


def test_measurement_stop_strategy_stable_window_unstable():
    strategy = StableWindowMeasuringStopStrategy(window_size=10, stability_percentage=90)
    assert not strategy.should_stop([new_event(1, 0)])

    # unstable measurement
    for i in range(8):
        assert not strategy.should_stop([new_event(1, i, measurement_id=i)])

    # stable measurement
    for i in range(8, 17):
        assert not strategy.should_stop([new_event(1, 1, measurement_id=i)])

    assert strategy.should_stop([new_event(1, 1, measurement_id=17)])


@pytest.mark.parametrize(
    "window_size,stability_percentage",
    [
        (10, 90),
        (100, 99),
        (1000, 99.9),
    ],
)
def test_measurement_stop_strategy_stable_window_stability(window_size: int, stability_percentage: float):
    strategy = StableWindowMeasuringStopStrategy(window_size=window_size, stability_percentage=stability_percentage)

    assert not strategy.should_stop([new_event(1, 300e6, measurement_id=0)])

    for i in range(1, window_size):
        assert not strategy.should_stop([new_event(1, 100e6, measurement_id=i)])

    assert strategy.should_stop([new_event(1, 100e6, measurement_id=window_size)])


def test_profile_toy_model():
    profile_config = ProfilingConfig(batch_sizes=[2**n for n in range(1, 5)])

    model = Module(ToyTorchModel())
    dataset = model.samples()

    result = profile(model, dataset, profile_config)
    assert result.status == ProfilingStatus.Status.SUCCESS
    assert len(result.results.entries) > 0


def test_profile_toy_model_no_batching():
    profile_config = ProfilingConfig(batch_sizes=[2**n for n in range(1, 5)], batching=False)

    model = Module(ToyTorchModel())
    dataset = model.samples()

    result = profile(model, dataset, profile_config)
    assert result.status == ProfilingStatus.Status.SUCCESS
    assert len(result.results.entries) > 0


@pytest.mark.parametrize("batching", [True, False])
def test_profile_backend(batching: bool):
    model = ToyTorchModel()
    graph_spec = model.graph_spec(batch_sizes=[1])

    class MockBackend(TorchEagerBackend):
        def __init__(self, model):
            self.model = model

        def infer(self, *args, **kwargs):
            """The following function imitates adding something to cache each time it is called.

            If the profile is idempotent, the cache argument should always be empty i.e. any changes should be discarded.
            """
            assert len(kwargs["cache"]) == 0, "Cache should be empty"
            kwargs["cache"].append("not important, should be discarded")
            return self.model(*args)

        def describe(self):
            return "mock_backend"

    backend = MockBackend(model)
    profile_config = ProfilingConfig(batch_sizes=[1, 2], batching=batching)

    samples = [(model.inputs(batch_sizes=[1]), {"cache": []})]
    result = profile_backend(backend, "test_model", graph_spec, samples, profile_config)  # type: ignore

    assert result.status == ProfilingStatus.Status.SUCCESS

    # Verify both calls produced results
    assert len(result.results.entries) > 0
