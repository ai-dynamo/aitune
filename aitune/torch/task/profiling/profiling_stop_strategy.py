# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profiling stop strategies for profiling."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from aitune.torch.task.profiling.events import ProfilingResultEvent
from aitune.torch.task.profiling.metrics import is_throughput_saturated


class ProfilingStopStrategy(ABC):
    """Strategy when to stop profiling."""

    @abstractmethod
    def should_stop(self, results: list[ProfilingResultEvent]) -> bool:
        """Check if the profiling should be stopped.

        Args:
            results: List of profiling results.

        Returns:
            True if the profiling should be stopped, False otherwise.
        """
        pass


@dataclass
class AllSamplesProfilingStopStrategy(ProfilingStopStrategy):
    """Strategy to do not stop profiling until all samples are processed."""

    def should_stop(self, results: list[ProfilingResultEvent]) -> bool:
        """Check if the profiling should be stopped."""
        return False


@dataclass
class ThroughputSaturatedProfilingStopStrategy(ProfilingStopStrategy):
    """Strategy to stop profiling when throughput is saturated.

    Additionally, backoff policy is introduced, allowing for few more measurements after saturation is detected.
    This is useful to avoid stopping profiling too early, when the model is still not fully optimized.

    Args:
        throughput_cutoff_threshold: Threshold for throughput saturation.
        throughput_backoff_limit: Number of measurements after saturation is detected to continue profiling. 0 or negative value means no backoff.
    """

    throughput_cutoff_threshold: float
    throughput_backoff_limit: int = 0

    _best_results: list[ProfilingResultEvent] = field(default_factory=list)
    _backoff_counter: int = 0

    def should_stop(self, results: list[ProfilingResultEvent]) -> bool:
        """Check if the profiling should be stopped."""
        # is_saturated == False also means best result for now
        is_saturated = is_throughput_saturated(results, self.throughput_cutoff_threshold, self._best_results)

        # keeping only best throughput results
        if not is_saturated:
            self._best_results = results

        if self.throughput_backoff_limit <= 0:
            return is_saturated

        if is_saturated:
            if self._backoff_counter >= self.throughput_backoff_limit:
                return True  # stop profiling
            self._backoff_counter += 1
        else:
            # resetting backoff counter, maybe better results are coming
            self._backoff_counter = 0

        return False  # continue profiling
