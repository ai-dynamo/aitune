# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metrics for profiling."""

import numpy as np

from aitune.torch.task.profiling.events import ProfilingResultEvent
from aitune.utils import validation


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
    mean_execution_time = np.mean([result.execution_time for result in results])
    return 1e9 / mean_execution_time


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
    execution_time_per_second = get_mean_executions_per_second(results)
    return batch_size * execution_time_per_second
