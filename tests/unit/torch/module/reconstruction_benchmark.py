# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared reporting for the reconstruction baseline; timings are observations, never assertions.

For a controlled comparison, use the same idle runner and software environment on both revisions::

    pytest -s -n 0 tests/unit/torch/module/test_sample_metadata.py \
        tests/unit/torch/module/test_reconstruction_contract.py \
        -k 'test_flatten_unflatten_timing or test_restore_outputs_timing' \
        -o junit_family=legacy --junitxml=tensor-reconstruction-benchmark.xml

CI retains the serial report in pytorch-gpu-results. Parallel unit-suite timings are only smoke
coverage. Compare absolute microseconds and repeated samples before selecting an optimization;
these CPU contract measurements do not establish model latency or GPU throughput improvements.
"""

import gc
import os
import platform
from pathlib import Path
from statistics import median
from timeit import Timer

import torch


def record_timings(operations, gc_enabled, record_property):
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
