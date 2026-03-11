# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aitune.torch.task.find_max_batch_size import find_max_batch_size, get_throughput_per_batch_size
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
        measurement_stop_strategy=NumStepsMeasuringStopStrategy(num_steps=1),
        profiling_stop_strategy=AllSamplesProfilingStopStrategy(),
        batch_sizes=[1, 2, 4],
    )


def test_highest_throughput_strategy_find_max_batch_size(mock_profiling_config, torch_device, tmp_path):
    model = ToyTorchModel()
    sample = model.sample().to(torch_device)
    _, throughput, _ = find_max_batch_size(
        model, "test", model.graph_spec(), [((sample,), {})], mock_profiling_config, torch_device, tmp_path
    )
    assert throughput > 0


def test_get_throughput_per_batch_size(mock_profiling_config):
    # Create mock profiling events with different batch sizes and timings
    events = [
        ProfilingResultEvent(
            timestamp=1,
            execution_time=1e9,
            batch_size=1,
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
    assert throughput_per_batch_size[0][0] == 4  # Highest throughput should be batch size 4
    assert throughput_per_batch_size[1][0] == 2  # Second highest should be batch size 2
    assert throughput_per_batch_size[2][0] == 1  # Lowest should be batch size 1

    # Verify throughput values are positive
    assert throughput_per_batch_size[0][1] == pytest.approx(2)
    assert throughput_per_batch_size[1][1] == pytest.approx(1.333333)
    assert throughput_per_batch_size[2][1] == pytest.approx(1)
