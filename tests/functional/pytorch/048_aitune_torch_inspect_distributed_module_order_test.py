# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate rank-zero inspection ordering across distributed workers.

The pytest parent relaunches this file with torchrun. Workers discover modules in different orders and measure opposite
local timings, then verify that inspection selection consistently follows rank zero.
"""

# /// script
# docker_image = "nvcr.io/nvidia/pytorch:26.06-py3"
# scope = "always"
# allow_failure = false
# additional_tags = ["gpu/4"]
# ///

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

from aitune.torch import inspect
from aitune.torch.distributed import coordinator

WORLD_SIZE = 4
FEATURES = 8
INSPECTION_DELAY_SECONDS = 0.1
DISTRIBUTED_LAUNCH_TIMEOUT_SECONDS = 15 * 60


class TimedLinear(nn.Module):
    """Linear layer with an inspection-only delay."""

    def __init__(self, delay_seconds: float):
        """Initialize the layer with its rank-local delay."""
        super().__init__()
        self.linear = nn.Linear(FEATURES, FEATURES)
        self.delay_seconds = delay_seconds

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Delay the inspected call before running the tensor operation."""
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        return torch.relu(self.linear(value))


class InspectionOrderModel(nn.Module):
    """Two candidates whose local discovery and timing order differs between ranks."""

    def __init__(self, rank: int):
        """Make ``first`` rank zero's first and slow candidate, reversing both elsewhere."""
        super().__init__()
        first = TimedLinear(INSPECTION_DELAY_SECONDS if rank == 0 else 0.0)
        second = TimedLinear(0.0 if rank == 0 else INSPECTION_DELAY_SECONDS)
        if rank == 0:
            self.first = first
            self.second = second
        else:
            self.second = second
            self.first = first

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Run both inspection candidates in discovery order."""
        return self.second(self.first(value))


def _run_worker() -> None:
    """Inspect and select the same rank-zero candidate on every rank."""
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    try:
        assert dist.get_world_size() == WORLD_SIZE
        torch.manual_seed(0)
        rank = dist.get_rank()
        model = InspectionOrderModel(rank).eval().to(device)
        dataset = [torch.randn(FEATURES, device=device), torch.randn(FEATURES, device=device)]
        expected_child_order = ("first", "second") if rank == 0 else ("second", "first")
        with coordinator.raise_if_any_rank_fails("Validating rank-local module discovery order"):
            assert tuple(name for name, _ in model.named_children()) == expected_child_order

        modules_info = inspect(
            model,
            dataset,
            number_of_iterations=2,
            warmup_iterations=0,
            min_depth=1,
            max_depth=1,
        )
        all_paths = tuple(module.object_path for module in modules_info.get_modules())
        limited_modules = modules_info.get_modules(limit=1)
        ratio_modules = modules_info.get_modules(min_execution_ratio=0.5)
        limited_paths = tuple(module.object_path for module in limited_modules)
        ratio_paths = tuple(module.object_path for module in ratio_modules)
        coordinator.verify_equal((all_paths, limited_paths, ratio_paths), "inspection selection")
        with coordinator.raise_if_any_rank_fails("Validating filtered inspection selection"):
            assert all_paths == (".first", ".second")
            assert limited_paths == (".first",)
            assert ratio_paths == (".first",)
        coordinator.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _launch_workers() -> None:
    """Relaunch this functional test with four application-owned workers."""
    if torch.cuda.device_count() < WORLD_SIZE:
        raise RuntimeError(f"This functional test requires {WORLD_SIZE} visible GPUs")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={WORLD_SIZE}",
            str(Path(__file__).resolve()),
        ],
        check=True,
        timeout=DISTRIBUTED_LAUNCH_TIMEOUT_SECONDS,
    )


def test_distributed_inspection_selection_uses_rank_zero_order() -> None:
    """Run the parent launcher or the rank-local functional workflow."""
    if "LOCAL_RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        _run_worker()
    else:
        _launch_workers()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    test_distributed_inspection_selection_uses_rank_zero_order()
