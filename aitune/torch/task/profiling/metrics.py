# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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
"""Metrics for profiling."""

import numpy as np

from aitune.torch.task.profiling.events import ProfilingResultEvent


def is_throughput_saturated(
    profiling_result: list[ProfilingResultEvent],
    throughput_cutoff_threshold: float,
    prev_results: list[ProfilingResultEvent],
) -> bool:
    """Validate if throughput saturated between consecutive samples.

    Args:
        profiling_result: Current profiling results.
        throughput_cutoff_threshold: Threshold for throughput saturation.
        prev_results: Previous profiling results or previous throughput.

    Returns:
        True when throughput saturated between consecutive samples. False when verification disabled or not yet saturated.
    """
    if not prev_results:
        return False

    prev_throughput = get_throughput(prev_results)
    new_throughput = get_throughput(profiling_result)

    # new_throughput minus few percent should be less than prev_throughput, then it is saturated
    return new_throughput * (1 - throughput_cutoff_threshold) < prev_throughput


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
