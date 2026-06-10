# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from aitune.torch.task.profiling.profiling_stop_strategy import (
    ThroughputSaturatedProfilingStopStrategy,
)
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


def test_profiling_config_materializes_batch_size_generator():
    batch_sizes = (2**n for n in range(3))

    profiling_config = ProfilingConfig(batch_sizes=batch_sizes)

    assert profiling_config.batch_sizes == [1, 2, 4]


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


@pytest.mark.parametrize("min_throughput_gain_ratio", [-0.01, 1.01])
def test_is_throughput_saturated_rejects_invalid_min_throughput_gain_ratio(min_throughput_gain_ratio: float):
    profiling_result = [
        new_event(1, 100e6),
        new_event(1, 100e6),
    ]

    with pytest.raises(ValueError, match="value must be between 0 and 1"):
        is_throughput_saturated(profiling_result, min_throughput_gain_ratio, profiling_result)


def test_throughput_saturated_threshold_strategy():
    strategy = ThroughputSaturatedProfilingStopStrategy(
        min_throughput_gain_ratio=0.01,
        throughput_backoff_limit=0,
    )
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


def test_profiling_config_default_profiling_stop_strategy():
    config = ProfilingConfig(batch_sizes=[1])

    assert isinstance(config.profiling_stop_strategy, ThroughputSaturatedProfilingStopStrategy)
    assert config.profiling_stop_strategy.min_throughput_gain_ratio == 0.05
    assert config.profiling_stop_strategy.throughput_backoff_limit == 2


def test_throughput_saturated_strategy_defaults():
    strategy = ThroughputSaturatedProfilingStopStrategy()

    assert strategy.min_throughput_gain_ratio == 0.05
    assert strategy.throughput_backoff_limit == 2


@pytest.mark.parametrize("min_throughput_gain_ratio", [-0.01, 1.01])
def test_throughput_saturated_strategy_rejects_invalid_min_throughput_gain_ratio(
    min_throughput_gain_ratio: float,
):
    with pytest.raises(ValueError, match="value must be between 0 and 1"):
        ThroughputSaturatedProfilingStopStrategy(min_throughput_gain_ratio=min_throughput_gain_ratio)


def test_throughput_saturated_strategy_rejects_negative_backoff_limit():
    with pytest.raises(ValueError, match="value must not be negative - greater than or equal to 0"):
        ThroughputSaturatedProfilingStopStrategy(throughput_backoff_limit=-1)


@pytest.mark.parametrize(
    "events_stop,min_throughput_gain_ratio,throughput_backoff_limit",
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
    events_stop, min_throughput_gain_ratio, throughput_backoff_limit
):
    strategy = ThroughputSaturatedProfilingStopStrategy(
        min_throughput_gain_ratio=min_throughput_gain_ratio, throughput_backoff_limit=throughput_backoff_limit
    )

    for events, should_stop in events_stop:
        # setup events
        events = [new_event(batch_size, execution_time) for batch_size, execution_time in events]

        # check if should stop
        assert strategy.should_stop(events) == should_stop, f"Events: {events}, should_stop: {should_stop}"

    assert should_stop, "All events should be consumed"


def test_measurement_stop_strategy_num_steps():
    strategy = NumStepsMeasuringStopStrategy(num_steps=10, warmup_samples=1)
    assert strategy.warmup_samples == 1
    assert not strategy.should_stop([new_event(1, 100e6)])
    for i in range(9):
        assert not strategy.should_stop([new_event(1, 100e6, measurement_id=i)])
    assert strategy.should_stop([new_event(1, 100e6)])
    assert strategy.should_stop([new_event(1, 100e6)])


def test_measurement_stop_strategy_num_steps_defaults():
    strategy = NumStepsMeasuringStopStrategy()

    assert strategy.num_steps == 20
    assert strategy.warmup_samples == 10


def test_measurement_stop_strategy_num_steps_warmup_samples():
    strategy = NumStepsMeasuringStopStrategy(num_steps=3, warmup_samples=2)
    events = [new_event(1, 100e6, measurement_id=i) for i in range(5)]

    assert not strategy.should_stop([events[0]])
    assert not strategy.should_stop([events[1]])
    assert not strategy.should_stop([events[2]])
    assert not strategy.should_stop([events[3]])
    assert strategy.should_stop([events[4]])
    assert strategy.get_events(events) == events[-3:]


def test_measurement_stop_strategy_num_steps_stops_after_warmup_and_steps():
    strategy = NumStepsMeasuringStopStrategy(num_steps=3, warmup_samples=2)
    events = [new_event(1, 100e6, measurement_id=i) for i in range(5)]

    assert not strategy.should_stop(events[:2])
    assert strategy.should_stop(events[2:])


def test_measurement_stop_strategy_num_steps_get_events_returns_last_steps_after_stop():
    strategy = NumStepsMeasuringStopStrategy(num_steps=2, warmup_samples=2)
    events = [new_event(1, 100e6, measurement_id=i) for i in range(5)]

    assert strategy.should_stop(events)
    assert strategy.get_events(events) == events[-2:]


@pytest.mark.parametrize("warmup_samples", [0, -1])
def test_measurement_stop_strategy_num_steps_rejects_non_positive_warmup_samples(warmup_samples: int):
    with pytest.raises(ValueError, match="value must be positive - greater than 0"):
        NumStepsMeasuringStopStrategy(warmup_samples=warmup_samples)


def test_measurement_stop_strategy_stable_window_defaults():
    strategy = StableWindowMeasuringStopStrategy()

    assert strategy.window_size == 20
    assert strategy.max_cv_ratio == 0.10
    assert strategy.warmup_samples == 10
    assert strategy.max_samples == 100


@pytest.mark.parametrize("max_cv_ratio", [-0.01, 1.01])
def test_measurement_stop_strategy_stable_window_rejects_invalid_max_cv_ratio(max_cv_ratio: float):
    with pytest.raises(ValueError, match="value must be between 0 and 1"):
        StableWindowMeasuringStopStrategy(max_cv_ratio=max_cv_ratio)


def test_measurement_stop_strategy_stable_window():
    strategy = StableWindowMeasuringStopStrategy(window_size=10, max_cv_ratio=0.90, warmup_samples=10, max_samples=100)

    for i in range(9):
        assert not strategy.should_stop([new_event(1, 100e6, measurement_id=i)])
    assert len(strategy._window) == 9
    assert not strategy.should_stop([new_event(1, 100e6, measurement_id=9)])

    for i in range(10, 19):
        assert not strategy.should_stop([new_event(1, 100e6, measurement_id=i)])

    assert strategy.should_stop([new_event(1, 100e6, measurement_id=19)])
    assert len(strategy._window) == 10
    assert strategy.should_stop([new_event(1, 100e6, measurement_id=20)])


def test_measurement_stop_strategy_stable_window_unstable():
    strategy = StableWindowMeasuringStopStrategy(window_size=10, max_cv_ratio=0.90, warmup_samples=10, max_samples=100)

    assert not strategy.should_stop([new_event(1, 300e6, measurement_id=0)])

    for i in range(1, 10):
        assert not strategy.should_stop([new_event(1, 100e6, measurement_id=i)])

    for i in range(10, 19):
        assert not strategy.should_stop([new_event(1, 100e6, measurement_id=i)])

    assert strategy.should_stop([new_event(1, 100e6, measurement_id=19)])


def test_measurement_stop_strategy_stable_window_unstable_when_not_enough_data():
    strategy = StableWindowMeasuringStopStrategy()

    for i in range(9):
        assert not strategy.should_stop([new_event(1, 100e6, measurement_id=i)])


def test_measurement_stop_strategy_stable_window_rejects_max_samples_below_window_size():
    with pytest.raises(
        ValueError,
        match="max_samples must be greater than or equal to window_size",
    ):
        StableWindowMeasuringStopStrategy(max_samples=9)


def test_measurement_stop_strategy_stable_window_rejects_default_max_samples_below_larger_window():
    with pytest.raises(
        ValueError,
        match="max_samples must be greater than or equal to window_size",
    ):
        StableWindowMeasuringStopStrategy(window_size=101)


def test_measurement_stop_strategy_stable_window_unstable_when_data_exceeds_max_cv_ratio():
    strategy = StableWindowMeasuringStopStrategy(window_size=10, max_cv_ratio=0.05, warmup_samples=10)

    for i in range(10):
        assert not strategy.should_stop([new_event(1, 1, measurement_id=i)])

    for i in range(10, 19):
        assert not strategy.should_stop([new_event(1, 1, measurement_id=i)])

    assert not strategy.should_stop([new_event(1, 5, measurement_id=19)])


def test_measurement_stop_strategy_stable_window_raises_when_max_samples_exhausted():
    strategy = StableWindowMeasuringStopStrategy(window_size=2, max_cv_ratio=0.01, warmup_samples=10, max_samples=4)

    for i in range(13):
        assert not strategy.should_stop([new_event(1, 1 if i % 2 == 0 else 3, measurement_id=i)])

    with pytest.raises(RuntimeError, match=r"Unable to collect stable results\."):
        strategy.should_stop([new_event(1, 3, measurement_id=13)])


def test_measurement_stop_strategy_stable_window_counts_result_events_as_samples():
    strategy = StableWindowMeasuringStopStrategy(window_size=2, max_cv_ratio=0.01, warmup_samples=10, max_samples=4)

    assert not strategy.should_stop([new_event(1, 1 if i % 2 == 0 else 3, measurement_id=i) for i in range(10)])
    assert not strategy.should_stop([new_event(1, 1 if i % 2 == 0 else 3, measurement_id=i) for i in range(10, 13)])
    with pytest.raises(RuntimeError, match=r"Unable to collect stable results\."):
        strategy.should_stop([new_event(1, 3, measurement_id=13)])


def test_measurement_stop_strategy_stable_window_stops_at_threshold_boundary():
    strategy = StableWindowMeasuringStopStrategy(window_size=2, max_cv_ratio=0.50, warmup_samples=10, max_samples=10)

    for i in range(10):
        assert not strategy.should_stop([new_event(1, i, measurement_id=i)])

    assert not strategy.should_stop([new_event(1, 1, measurement_id=10)])
    assert strategy.should_stop([new_event(1, 3, measurement_id=11)])


def test_measurement_stop_strategy_stable_window_does_not_stop_above_threshold():
    strategy = StableWindowMeasuringStopStrategy(window_size=2, max_cv_ratio=0.50, warmup_samples=10, max_samples=10)

    for i in range(10):
        assert not strategy.should_stop([new_event(1, i, measurement_id=i)])

    assert not strategy.should_stop([new_event(1, 1, measurement_id=10)])
    assert not strategy.should_stop([new_event(1, 3.1, measurement_id=11)])


def test_measurement_stop_strategy_stable_window_get_events_returns_last_window():
    strategy = StableWindowMeasuringStopStrategy(window_size=3, max_samples=10)
    events = [new_event(1, 100e6, measurement_id=i) for i in range(5)]

    assert strategy.get_events(events) == events[-3:]


@pytest.mark.parametrize(
    "window_size,max_cv_ratio",
    [
        (10, 0.05),
        (100, 0.01),
        (1000, 0.001),
    ],
)
def test_measurement_stop_strategy_stable_window_stability(window_size: int, max_cv_ratio: float):
    strategy = StableWindowMeasuringStopStrategy(
        window_size=window_size,
        max_cv_ratio=max_cv_ratio,
        warmup_samples=10,
        max_samples=2 * window_size,
    )

    for i in range(10):
        assert not strategy.should_stop([new_event(1, 300e6, measurement_id=i)])

    for i in range(10, window_size + 9):
        assert not strategy.should_stop([new_event(1, 100e6, measurement_id=i)])

    assert strategy.should_stop([new_event(1, 100e6, measurement_id=window_size + 9)])


def test_profile_toy_model():
    profile_config = ProfilingConfig(
        batch_sizes=[2**n for n in range(1, 5)],
        measurement_stop_strategy=NumStepsMeasuringStopStrategy(num_steps=1, warmup_samples=1),
    )

    model = Module(ToyTorchModel())
    dataset = model.samples()

    result = profile(model, dataset, profile_config)
    assert result.status == ProfilingStatus.Status.SUCCESS
    assert len(result.results.entries) > 0


def test_profile_toy_model_no_batching():
    profile_config = ProfilingConfig(
        batch_sizes=[2**n for n in range(1, 5)],
        batching=False,
        measurement_stop_strategy=NumStepsMeasuringStopStrategy(num_steps=1, warmup_samples=1),
    )

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

            If the profile is idempotent, the cache argument should always be empty
            i.e. any changes should be discarded.
            """
            assert len(kwargs["cache"]) == 0, "Cache should be empty"
            kwargs["cache"].append("not important, should be discarded")
            return self.model(*args)

        def describe(self):
            return "mock_backend"

    backend = MockBackend(model)
    profile_config = ProfilingConfig(
        batch_sizes=[1, 2],
        batching=batching,
        measurement_stop_strategy=NumStepsMeasuringStopStrategy(num_steps=1, warmup_samples=1),
    )

    samples = [(model.inputs(batch_sizes=[1]), {"cache": []})]
    result = profile_backend(backend, "test_model", graph_spec, samples, profile_config)  # type: ignore

    assert result.status == ProfilingStatus.Status.SUCCESS

    # Verify both calls produced results
    assert len(result.results.entries) > 0
