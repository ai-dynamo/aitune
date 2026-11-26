# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report CPU microbenchmark samples and environment details without latency thresholds."""

import gc
import os
import platform
from pathlib import Path
from statistics import median
from timeit import Timer

import torch


def record_timings(operations, gc_enabled, record_property):
    """Measure CPU microbenchmark operations and record per-call timings.

    Warm up each operation, then time repeated batches of calls. Each sample is the
    batch's average time per call in microseconds. Record the individual samples,
    their median, and the benchmark environment as pytest properties. Also print
    the results for inspection. Timings are informational and have no pass/fail threshold.

    Args:
        operations: Mapping from operation names to zero-argument callables. Each callable
            is reused throughout warmup and measurement.
        gc_enabled: Whether to enable garbage collection during warmup and measurement.
            The caller's original GC state is restored even if an operation fails.
        record_property: Pytest fixture that records a named value in the test report.
    """
    iterations, repeats, warmup = 100, 5, 20
    processor = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        processor = next(
            (
                line.partition(":")[2].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            processor,
        )
    environment = {
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": processor,
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "runner": os.environ.get("RUNNER_NAME", "unspecified"),
        "commit": os.environ.get("GITHUB_SHA", "unspecified"),
        "gc_enabled": gc_enabled,
        "gc_thresholds": gc.get_threshold(),
        "iterations": iterations,
        "repeats": repeats,
        "warmup": warmup,
    }
    record_property("benchmark_environment", environment)
    initially_enabled = gc.isenabled()
    try:
        for operation, function in operations.items():
            # timeit disables GC before setup. Explicitly select both modes here, and restore
            # the caller's GC state even if setup, warmup, or a measured call fails.
            timer = Timer(function, setup=gc.enable if gc_enabled else gc.disable)
            timer.timeit(number=warmup)
            samples = [elapsed * 1e6 / iterations for elapsed in timer.repeat(repeat=repeats, number=iterations)]
            record_property(f"{operation}_median_us", median(samples))
            record_property(f"{operation}_samples_us", samples)
            print(f"{operation}: {median(samples):.3f} us/call; samples={samples}; environment={environment}")
    finally:
        (gc.enable if initially_enabled else gc.disable)()
