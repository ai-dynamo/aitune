# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify that four local GPUs can form an NCCL process group and all-reduce."""

import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist


WORLD_SIZE = 4
TIMEOUT_SECONDS = 300


def _run_worker() -> None:
    """Create the rank-local NCCL communicator and verify one collective."""
    print(f"Running worker on rank {dist.get_rank()}", flush=True)
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    print(f"CUDA device set to {device}", flush=True)
    dist.init_process_group("nccl", device_id=device)
    print(f"NCCL initialized on rank {dist.get_rank()}", flush=True)
    try:
        print(f"NCCL world size: {dist.get_world_size()}", flush=True)
        assert dist.get_world_size() == WORLD_SIZE
        value = torch.tensor(float(dist.get_rank()), device=device)
        dist.all_reduce(value)
        print(f"NCCL all_reduce completed on rank {dist.get_rank()}", flush=True)
        torch.cuda.synchronize(device)
        print(f"NCCL synchronization completed on rank {dist.get_rank()}", flush=True)
        assert value.item() == sum(range(WORLD_SIZE))
        print(f"NCCL sanity passed on rank {dist.get_rank()}", flush=True)
    finally:
        dist.destroy_process_group()
        print(f"NCCL process group destroyed on rank {dist.get_rank()}", flush=True)


def main() -> None:
    """Run one worker per visible GPU, or execute the current torchrun worker."""
    if "LOCAL_RANK" in os.environ:
        _run_worker()
        return
    if torch.cuda.device_count() < WORLD_SIZE:
        raise RuntimeError(f"Expected {WORLD_SIZE} visible GPUs, found {torch.cuda.device_count()}")
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
        timeout=TIMEOUT_SECONDS,
    )


if __name__ == "__main__":
    main()
