# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metrics for profiling."""

from dataclasses import dataclass
from statistics import geometric_mean

import numpy as np

from aitune.torch.task.profiling.events import ProfilingResultEvent
from aitune.utils import validation


@dataclass(frozen=True)
class BoundaryLatencies:
    """Mean latencies in milliseconds at the two shapes of a single batch.

    Static shapes have equal values. Combined scores are internal ranking values;
    callers report min_shape and use it for the existing eager speedup output.
    """

    min_shape: float
    max_shape: float

    @classmethod
    def from_samples(cls, latencies: tuple[float, ...]) -> "BoundaryLatencies":
        """Interpret the boundary workflow's primary sample and optional maximum sample."""
        if len(latencies) not in (1, 2):
            raise ValueError("Boundary scoring requires one static sample or two boundary samples")
        return cls(latencies[0], latencies[-1])

    @property
    def latency_score(self) -> float:
        """Return the geometric-mean latency used to rank candidates; lower is better."""
        return geometric_mean((self.min_shape, self.max_shape))

    def throughput_score(self, batch_size: int) -> float:
        """Return geometric-mean throughput for ranking; higher is better.

        Both shapes use the same batch size, so the geometric mean of their
        throughputs equals batch_size / geometric_mean(latencies). Convert the
        millisecond latencies to samples per second.
        """
        return batch_size * 1000 / self.latency_score


def is_throughput_saturated(
    profiling_result: list[ProfilingResultEvent],
    min_throughput_gain_ratio: float,
    prev_results: list[ProfilingResultEvent],
) -> bool:
    """Validate if throughput saturated between consecutive samples.

    Args:
        profiling_result: Current profiling results.
        min_throughput_gain_ratio: Minimum relative throughput gain required to continue profiling.
        prev_results: Previous profiling results or previous throughput.

    Returns:
        True when throughput saturated between consecutive samples.
        False when verification disabled or not yet saturated.
    """
    validation.in_range(min_throughput_gain_ratio, min_value=0, max_value=1)

    if not prev_results:
        return False

    prev_throughput = get_throughput(prev_results)
    new_throughput = get_throughput(profiling_result)

    # new_throughput minus required gain should be less than prev_throughput, then it is saturated
    return new_throughput * (1 - min_throughput_gain_ratio) < prev_throughput


def get_batch_size(results: list[ProfilingResultEvent]) -> int:
    """Get batch size from profiling results.

    Args:
        results: List of profiling results.

    Returns:
        Batch size.
    """
    batch_sizes = [result.batch_size for result in results]
    if len(set(batch_sizes)) != 1:
        raise RuntimeError("All batch sizes must be the same")
    return batch_sizes[0]


def get_mean_executions_per_second(results: list[ProfilingResultEvent]) -> float:
    """Get execution time per second from profiling results.

    Args:
        results: List of profiling results.

    Returns:
        Executions per second based on mean execution time.
    """
    mean_execution_time = np.mean([result.execution_time for result in results], dtype=np.float64)
    return float(1e9 / mean_execution_time)


def get_latency(results: list[ProfilingResultEvent]) -> float:
    """Get mean latency from profiling results.

    Args:
        results: List of profiling results.

    Returns:
        Mean latency in milliseconds.
    """
    mean_execution_time_ns = np.mean([result.execution_time for result in results])
    return float(mean_execution_time_ns) / 1e6


def get_throughput(results: list[ProfilingResultEvent], batch_size: int | None = None) -> float:
    """Get throughput from profiling results.

    Args:
        results: List of profiling results.
        batch_size: Batch size to use for throughput calculation. If not provided, it will be inferred from the results.

    Returns:
        Throughput in samples per second.
    """
    batch_size = batch_size or get_batch_size(results)
    return batch_size * get_mean_executions_per_second(results)


def throughput_speedup(
    candidate_latency: float, eager_latency: float, *, candidate_batch_size: int, eager_batch_size: int
) -> float:
    """Compare throughput at each backend's selected batch size.

    Both latencies use the same units, so the conversion to seconds cancels out.
    """
    candidate_throughput = candidate_batch_size / candidate_latency
    eager_throughput = eager_batch_size / eager_latency
    return candidate_throughput / eager_throughput


def aggregate_throughput_per_batch_size(
    gathered_throughputs: list[dict[int, float]],
) -> list[tuple[int, float]]:
    """Rank scalar batch-throughput scores by the worst rank, breaking ties by batch size.

    Args:
        gathered_throughputs: One batch-size-to-throughput mapping per rank. Each
            rank must have measured the same batch sizes.

    Returns:
        Batch size and worst-rank throughput pairs, ordered by decreasing
        throughput and then increasing batch size.

    Raises:
        RuntimeError: If ranks report different sets of batch sizes.
    """
    batch_sizes = set(gathered_throughputs[0])
    if any(set(candidate) != batch_sizes for candidate in gathered_throughputs[1:]):
        details = ", ".join(f"rank {rank}: {sorted(candidate)}" for rank, candidate in enumerate(gathered_throughputs))
        raise RuntimeError(f"Distributed profiled batch sizes differ across ranks: {details}")

    aggregated = [
        (batch_size, min(candidate[batch_size] for candidate in gathered_throughputs)) for batch_size in batch_sizes
    ]
    return sorted(aggregated, key=lambda result: (-result[1], result[0]))
