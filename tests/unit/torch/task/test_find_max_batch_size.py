# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aitune.torch.task.find_max_batch_size import (
    _aggregate_throughput_per_batch_size,
    find_max_batch_size,
    get_throughput_per_batch_size,
)
from aitune.torch.task.profiling import (
    AllSamplesProfilingStopStrategy,
    ModelExecutionTimeMeasuringStrategy,
    NumStepsMeasuringStopStrategy,
    ProfilingConfig,
)
from aitune.torch.task.profiling.events import ProfilingResultEvent
from tests.toy_models.torch_models import ToyTorchModel


@pytest.fixture
def mock_profiling_config():
    return ProfilingConfig(
        measuring_strategy=ModelExecutionTimeMeasuringStrategy(),
        measurement_stop_strategy=NumStepsMeasuringStopStrategy(num_steps=1, warmup_samples=1),
        profiling_stop_strategy=AllSamplesProfilingStopStrategy(),
        batch_sizes=[1, 2, 4],
    )


def test_max_throughput_strategy_find_max_batch_size(mock_profiling_config, torch_device, tmp_path):
    model = ToyTorchModel()
    samples = model.sample_store(tmp_path, batch_sizes=[1], device=torch_device)
    _, throughput, _ = find_max_batch_size(
        model, "test", model.graph_spec(), samples, mock_profiling_config, torch_device, tmp_path
    )
    assert throughput > 0


def test_get_throughput_per_batch_size(mock_profiling_config):
    # Create mock profiling events with different batch sizes and timings
    events = [
        ProfilingResultEvent(
            timestamp=0,
            execution_time=10e9,
            batch_size=1,
            phase="inference",
            model_name="test",
            backend_details="test",
        ),
        ProfilingResultEvent(
            timestamp=1,
            execution_time=1e9,
            batch_size=1,
            phase="inference",
            model_name="test",
            backend_details="test",
        ),
        ProfilingResultEvent(
            timestamp=1,
            execution_time=10e9,
            batch_size=2,
            phase="inference",
            model_name="test",
            backend_details="test",
        ),
        ProfilingResultEvent(
            timestamp=2,
            execution_time=1.5e9,
            batch_size=2,
            phase="inference",
            model_name="test",
            backend_details="test",
        ),
        ProfilingResultEvent(
            timestamp=2,
            execution_time=10e9,
            batch_size=4,
            phase="inference",
            model_name="test",
            backend_details="test",
        ),
        ProfilingResultEvent(
            timestamp=3,
            execution_time=2e9,
            batch_size=4,
            phase="inference",
            model_name="test",
            backend_details="test",
        ),
    ]

    # Get throughput per batch size
    throughput_per_batch_size = get_throughput_per_batch_size(events, mock_profiling_config.measurement_stop_strategy)

    # Verify results
    assert len(throughput_per_batch_size) == 3
    assert throughput_per_batch_size[0][0] == 4  # Maximum throughput should be batch size 4
    assert throughput_per_batch_size[1][0] == 2  # Second highest should be batch size 2
    assert throughput_per_batch_size[2][0] == 1  # Lowest should be batch size 1

    # Verify throughput values are positive
    assert throughput_per_batch_size[0][1] == pytest.approx(2)
    assert throughput_per_batch_size[1][1] == pytest.approx(1.333333)
    assert throughput_per_batch_size[2][1] == pytest.approx(1)


def test_aggregate_throughput_uses_worst_rank_and_shared_optimum():
    results = _aggregate_throughput_per_batch_size([
        {1: 10.0, 2: 20.0},
        {1: 9.0, 2: 8.0},
    ])

    assert results == [(1, 9.0), (2, 8.0)]


def test_aggregate_throughput_rejects_different_profiled_batches():
    with pytest.raises(RuntimeError, match="profiled batch sizes differ"):
        _aggregate_throughput_per_batch_size([
            {1: 10.0, 2: 20.0},
            {1: 9.0},
        ])
