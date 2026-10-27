# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# scope = "always"
# ///

"""Validate that kernel profiling does not slow down later inference in the same process."""

import logging
import time
from logging import basicConfig

import torch
from torch import nn

from aitune.torch import ModuleFunctionKernelProfiler

logger = logging.getLogger(__name__)

MIN_THROUGHPUT_RATIO = 0.9
NUM_ITERATIONS = 200
NUM_REPEATS = 5


def _launch_bound_model() -> nn.Module:
    """Many small kernels make CPU launch overhead, and so CUPTI overhead, visible."""
    layers = []
    for _ in range(16):
        layers += [nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.Hardswish()]
    return nn.Sequential(*layers).to("cuda", torch.float16).eval()


@torch.no_grad()
def _throughput(model, sample: torch.Tensor) -> float:
    """Return the best samples/second of several timed repeats."""
    for _ in range(NUM_ITERATIONS):
        model(sample)
    best = 0.0
    for _ in range(NUM_REPEATS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(NUM_ITERATIONS):
            model(sample)
        torch.cuda.synchronize()
        best = max(best, NUM_ITERATIONS * sample.shape[0] / (time.perf_counter() - start))
    return best


def _check_profiling_keeps_throughput(name: str, compile_mode: str | None):
    model = _launch_bound_model()
    sample = torch.randn(1, 16, 32, 32, device="cuda", dtype=torch.float16)
    runnable = torch.compile(model, mode=compile_mode) if compile_mode else model

    before = _throughput(runnable, sample)
    ModuleFunctionKernelProfiler().profile(model, [((sample,), {})])
    after = _throughput(runnable, sample)

    ratio = after / before
    logger.info("%s: before %.1f samples/second, after %.1f samples/second, ratio %.3f", name, before, after, ratio)
    assert ratio >= MIN_THROUGHPUT_RATIO, f"{name}: throughput after profiling dropped to {ratio:.3f} of baseline"


def test_profiling_keeps_eager_throughput():
    _check_profiling_keeps_throughput("eager", compile_mode=None)


def test_profiling_keeps_cuda_graph_throughput():
    _check_profiling_keeps_throughput("inductor reduce-overhead", compile_mode="reduce-overhead")


if __name__ == "__main__":
    basicConfig(level=logging.INFO, format="%(message)s", force=True)
    test_profiling_keeps_eager_throughput()
    test_profiling_keeps_cuda_graph_throughput()
