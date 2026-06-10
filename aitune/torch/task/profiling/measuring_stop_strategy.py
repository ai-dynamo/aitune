# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measuring stop strategies for profiling."""

from abc import ABC, abstractmethod

import numpy as np

from aitune.torch.task.profiling.events import ProfilingResultEvent
from aitune.utils import validation


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

    def __init__(self, num_steps: int = 20, warmup_samples: int = 10):
        """Initialize the measuring stop strategy.

        Args:
            num_steps: Minimal number of steps for performing the measurement. Defaults to 20.
            warmup_samples: Number of initial samples excluded from measurement. Defaults to 10.
        """
        self.num_steps = num_steps
        self.warmup_samples = warmup_samples

        validation.positive(self.num_steps)
        validation.positive(self.warmup_samples)

        self._samples_seen = 0
        self._min_samples_seen = self.warmup_samples + self.num_steps

    def should_stop(self, results: list[ProfilingResultEvent]) -> bool:
        """Check if the measurement should be stopped."""
        self._samples_seen += len(results)
        return self._samples_seen >= self._min_samples_seen

    def get_events(self, results: list[ProfilingResultEvent]) -> list[ProfilingResultEvent]:
        """Get final measured events after should_stop() has returned True."""
        return results[-self.num_steps :]


class StableWindowMeasuringStopStrategy(MeasuringStopStrategy):
    """Strategy to stop collecting measurements after a stable sliding window has been collected."""

    def __init__(
        self,
        window_size: int = 20,
        max_cv_ratio: float = 0.10,
        warmup_samples: int = 10,
        max_samples: int = 100,
    ):
        """Initialize the measuring stop strategy.

        Args:
            window_size: Number of samples in the stability window. Defaults to 20.
            max_cv_ratio: Maximum allowed coefficient of variation for latency, expressed as a ratio.
                Defaults to 0.10.
            warmup_samples: Number of initial samples excluded from stability evaluation. Defaults to 10.
            max_samples: Maximum measured non-warmup samples before failing. Defaults to 100.
        """
        self.window_size = window_size
        self.max_cv_ratio = max_cv_ratio
        self.warmup_samples = warmup_samples
        self.max_samples = max_samples

        validation.ratio(self.max_cv_ratio)
        validation.positive(self.window_size)
        validation.positive(self.max_samples)
        validation.positive(self.warmup_samples)

        if self.max_samples < self.window_size:
            raise ValueError("max_samples must be greater than or equal to window_size.")

        self._window: list[ProfilingResultEvent] = []
        self._samples_seen = 0

        self._min_samples_seen = self.window_size + self.warmup_samples
        self._max_samples_seen = self.max_samples + self.warmup_samples

    def should_stop(self, results: list[ProfilingResultEvent]) -> bool:
        """Check if the measurement should be stopped."""
        self._window += results
        self._samples_seen += len(results)

        if self._samples_seen < self._min_samples_seen:
            return False

        self._window = self._window[-self.window_size :]
        execution_times = [result.execution_time for result in self._window]
        avg_latency = np.mean(execution_times)
        cv_ratio = np.std(execution_times) / avg_latency

        if cv_ratio <= self.max_cv_ratio:
            return True

        if self._samples_seen >= self._max_samples_seen:
            current_cv_ratio = float(cv_ratio)
            raise RuntimeError(
                "Unable to collect stable results. "
                f"Current CV of last window: {current_cv_ratio:.2%}. "
                f"Expected max CV: {self.max_cv_ratio:.2%}."
            )

        return False

    def get_events(self, results: list[ProfilingResultEvent]) -> list[ProfilingResultEvent]:
        """Get events to use for measuring stop strategy."""
        return results[-self.window_size :]
