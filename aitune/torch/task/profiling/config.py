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


@dataclass(init=False)
class ProfilingConfig:
    """Configuration for profiling.

    Each scan copies the profiling stop policy, and each boundary measurement
    copies the measurement stop policy. Their counters and windows are isolated;
    the measuring_strategy instance itself is reused.

    Attributes:
        batch_sizes: List of batch sizes to profile.
        batching: Whether profiling should batch samples together.
        measuring_strategy: Strategy to measure the model.
        measurement_stop_strategy: Strategy to stop collecting measurements.
        profiling_stop_strategy: Strategy to stop profiling.
    """

    batch_sizes: list[int]
    batching: bool
    measuring_strategy: MeasuringStrategy
    measurement_stop_strategy: MeasuringStopStrategy
    profiling_stop_strategy: ProfilingStopStrategy

    def __init__(
        self,
        batch_sizes: list[int] | Generator[int, None, None],
        batching: bool = True,
        measuring_strategy: MeasuringStrategy | None = None,
        measurement_stop_strategy: MeasuringStopStrategy | None = None,
        profiling_stop_strategy: ProfilingStopStrategy | None = None,
    ):
        """Resolve optional inputs once; constructed strategy fields are never None.

        Keep explicit None arguments compatible with the generated initializer
        previously used by this dataclass, including dataclasses.replace().
        """
        self.batch_sizes = list(batch_sizes) if isinstance(batch_sizes, Generator) else batch_sizes
        if not self.batch_sizes:
            raise ValueError("batch_sizes must be provided")
        self.batching = batching
        self.measuring_strategy = (
            measuring_strategy if measuring_strategy is not None else ModelExecutionTimeMeasuringStrategy()
        )
        self.measurement_stop_strategy = (
            measurement_stop_strategy if measurement_stop_strategy is not None else NumStepsMeasuringStopStrategy()
        )
        self.profiling_stop_strategy = (
            profiling_stop_strategy
            if profiling_stop_strategy is not None
            else ThroughputSaturatedProfilingStopStrategy()
        )
