# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measuring stop strategies for profiling."""

from abc import ABC, abstractmethod

import numpy as np

from aitune.torch.task.profiling.events import ProfilingResultEvent


class MeasuringStopStrategy(ABC):
    """Strategy when to stop collecting measurements for given batch size."""

    @abstractmethod
    def should_stop(self, results: list[ProfilingResultEvent]) -> bool:
        """Check if the measurement should be stopped.

        Args:
            results: List of profiling results events for a one measurement.

        Returns:
            True if the measurement should be stopped, False otherwise.
        """
        pass

    @abstractmethod
    def get_events(self, results: list[ProfilingResultEvent]) -> list[ProfilingResultEvent]:
        """Get events to use for measuring stop strategy."""
        pass


class NumStepsMeasuringStopStrategy(MeasuringStopStrategy):
    """Strategy to stop collecting measurements after a given number of steps."""

    def __init__(self, num_steps: int):
        """Initialize the measuring stop strategy."""
        self.num_steps = num_steps
        self._steps_counter = 0

    def should_stop(self, _: list[ProfilingResultEvent]) -> bool:
        """Check if the measurement should be stopped."""
        self._steps_counter += 1
        return self._steps_counter >= self.num_steps

    def get_events(self, results: list[ProfilingResultEvent]) -> list[ProfilingResultEvent]:
        """Get events to use for measuring stop strategy."""
        return results[-self.num_steps :]


class StableWindowMeasuringStopStrategy(MeasuringStopStrategy):
    """Strategy to stop collecting measurements after stable windows of measurements have been collected."""

    def __init__(
        self,
        window_size: int,
        stability_percentage: float,
    ):
        """Initialize the measuring stop strategy."""
        self.window_size = window_size
        self.stability_percentage = stability_percentage
        self._window: list[ProfilingResultEvent] = []
        self._samples_seen = 0
        self._warmup_samples = window_size

    def should_stop(self, results: list[ProfilingResultEvent]) -> bool:
        """Check if the measurement should be stopped."""
        self._window += results
        self._samples_seen += len(results)

        if self._samples_seen < self._warmup_samples + self.window_size:
            return False

        self._window = self._window[-self.window_size :]
        execution_times = [result.execution_time for result in self._window]
        avg_latency = np.mean(execution_times)
        deviation_perc = np.abs((execution_times - avg_latency) / avg_latency * 100)

        return np.all(deviation_perc < self.stability_percentage)

    def get_events(self, results: list[ProfilingResultEvent]) -> list[ProfilingResultEvent]:
        """Get events to use for measuring stop strategy."""
        return results[-self.window_size :]
