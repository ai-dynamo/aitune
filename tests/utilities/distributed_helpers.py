# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for functional tests launched with torchrun."""

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist

DISTRIBUTED_LAUNCH_TIMEOUT_SECONDS = 15 * 60


def run_or_launch_distributed_test(
    worker: Callable[[torch.device], None],
    script_path: str | Path,
    *,
    world_size: int,
) -> None:
    """Run a distributed worker or launch the calling test script with torchrun.

    Args:
        worker: Rank-local test function receiving its CUDA device.
        script_path: Test script relaunched in distributed workers.
        world_size: Number of workers and visible GPUs required by the test.
    """
    if "LOCAL_RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        _run_worker(worker, world_size)
    else:
        _launch_workers(script_path, world_size, DISTRIBUTED_LAUNCH_TIMEOUT_SECONDS)


def _run_worker(worker: Callable[[torch.device], None], world_size: int) -> None:
    """Initialize one NCCL worker, run the test, and destroy the process group."""
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    try:
        assert dist.get_world_size() == world_size
        worker(device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _launch_workers(script_path: str | Path, world_size: int, timeout_seconds: float) -> None:
    """Relaunch a functional test with one torchrun worker per required GPU."""
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(f"This functional test requires {world_size} visible GPUs")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={world_size}",
            str(Path(script_path).resolve()),
        ],
        check=True,
        timeout=timeout_seconds,
    )
