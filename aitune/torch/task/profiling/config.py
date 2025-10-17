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
"""Profiling configuration."""

from collections.abc import Generator
from dataclasses import dataclass

from aitune.torch.config import DEFAULT_THROUGHPUT_BACKOFF_LIMIT, DEFAULT_THROUGHPUT_CUTOFF_THRESHOLD
from aitune.torch.task.profiling.measuring_stop_strategy import MeasuringStopStrategy, NumStepsMeasuringStopStrategy
from aitune.torch.task.profiling.measuring_strategy import MeasuringStrategy, ModelExecutionTimeMeasuringStrategy
from aitune.torch.task.profiling.profiling_stop_strategy import (
    ProfilingStopStrategy,
    ThroughputSaturatedProfilingStopStrategy,
)


@dataclass
class ProfilingConfig:
    """Configuration for profiling.

    Note: all profiling configuration strategies are deepcopied before use.

    Attributes:
        batch_sizes: List of batch sizes to profile.
        measuring_strategy: Strategy to measure the model. Default is ModelExecutionTimeMeasuringStrategy().
        measurement_stop_strategy: Strategy to stop collecting measurements. Default is NumStepsMeasuringStopStrategy(num_steps=10).
        profiling_stop_strategy: Strategy to stop profiling. Default is ThroughputSaturatedProfilingStopStrategy(throughput_cutoff_threshold=0.05).
    """

    batch_sizes: list[int] | Generator[int, None, None]

    batching: bool = True
    measuring_strategy: MeasuringStrategy | None = None
    measurement_stop_strategy: MeasuringStopStrategy | None = None
    profiling_stop_strategy: ProfilingStopStrategy | None = None

    def __post_init__(self):
        """Post-init."""
        if not self.batch_sizes:
            raise ValueError("batch_sizes must be provided")

        if self.measuring_strategy is None:
            self.measuring_strategy = ModelExecutionTimeMeasuringStrategy()

        if self.measurement_stop_strategy is None:
            self.measurement_stop_strategy = NumStepsMeasuringStopStrategy(num_steps=10)

        if self.profiling_stop_strategy is None:
            self.profiling_stop_strategy = ThroughputSaturatedProfilingStopStrategy(
                throughput_cutoff_threshold=DEFAULT_THROUGHPUT_CUTOFF_THRESHOLD,
                throughput_backoff_limit=DEFAULT_THROUGHPUT_BACKOFF_LIMIT,
            )
