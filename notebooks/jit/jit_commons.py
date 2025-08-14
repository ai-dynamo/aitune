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
"""Common utilities for JIT."""

from logging import basicConfig
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

basicConfig(
    level="INFO", format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s", datefmt="%H:%M:%S", force=True
)


def visualize_latency(callback, iterations=2):
    """Visualize the latency of a callback.

    Args:
        callback: The callback to visualize the latency of.
        iterations: The number of iterations to run the callback.
    """
    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    latencies = measure_latencies(callback, iterations)
    plt.figure(figsize=(6, 4), dpi=100)

    # Plot all points in blue
    plt.scatter(range(len(latencies)), latencies, color="blue", label="Original")

    # Find the maximum latency point (tuning phase)
    max_idx = np.argmax(latencies)
    max_latency = latencies[max_idx]
    plt.scatter([max_idx], [max_latency], color="red", zorder=5, label="Tuning")

    # Plot points after maximum in green (tuned phase)
    if max_idx < len(latencies) - 1:
        after_max_indices = list(range(max_idx + 1, len(latencies)))
        after_max_latencies = [latencies[i] for i in after_max_indices]
        plt.scatter(after_max_indices, after_max_latencies, color="green", zorder=5, label="Tuned")

    plt.xlabel("Iteration")
    plt.ylabel("Latency (s)")
    plt.title("Latencies during iterations")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()


def measure_latencies(callback, iterations):
    """Measure the latencies of the callback.

    Args:
        callback: The callback to measure the latencies of.
        iterations: The number of iterations to run the callback.
    """
    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    latencies = []
    for _ in range(iterations):
        start = perf_counter()
        _ = callback()
        latencies.append(perf_counter() - start)
    return latencies


def benchmark(callback, iterations=2):
    """Benchmark the callback.

    Args:
        callback: The callback to benchmark.
        iterations: The number of iterations to run the callback.

    Returns:
        str: Formatted string with mean ± std and number of iterations.
    """
    latencies = measure_latencies(callback, iterations)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    return f"{mean_latency:.3}s ± {std_latency:.3} ({iterations} iterations)"
