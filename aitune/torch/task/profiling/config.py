# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profiling configuration."""

from collections.abc import Generator
from dataclasses import dataclass

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
        batching: Whether profiling should batch samples together.
        measuring_strategy: Strategy to measure the model.
        measurement_stop_strategy: Strategy to stop collecting measurements.
        profiling_stop_strategy: Strategy to stop profiling.
    """

    batch_sizes: list[int] | Generator[int, None, None]

    batching: bool = True
    measuring_strategy: MeasuringStrategy | None = None
    measurement_stop_strategy: MeasuringStopStrategy | None = None
    profiling_stop_strategy: ProfilingStopStrategy | None = None

    def __post_init__(self):
        """Post-init."""
        if isinstance(self.batch_sizes, Generator):
            self.batch_sizes = list(self.batch_sizes)

        if not self.batch_sizes:
            raise ValueError("batch_sizes must be provided")

        if self.measuring_strategy is None:
            self.measuring_strategy = ModelExecutionTimeMeasuringStrategy()

        if self.measurement_stop_strategy is None:
            self.measurement_stop_strategy = NumStepsMeasuringStopStrategy()

        if self.profiling_stop_strategy is None:
            self.profiling_stop_strategy = ThroughputSaturatedProfilingStopStrategy()
