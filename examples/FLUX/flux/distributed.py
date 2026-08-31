# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distributed helpers owned by the Flux example."""

import os
from pathlib import Path

import torch
import torch.distributed as dist


def initialize(enabled: bool) -> None:
    """Initialize one NCCL rank per GPU when context parallelism is enabled."""
    if not enabled:
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")


def is_rank_zero() -> bool:
    """Return whether this process owns user-visible output."""
    return not dist.is_initialized() or dist.get_rank() == 0


def synchronize() -> None:
    """Wait for every rank, or for local CUDA work outside distributed execution."""
    if dist.is_initialized():
        dist.barrier()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def shutdown() -> None:
    """Destroy the application-owned process group after normal or failed execution."""
    if dist.is_initialized():
        dist.destroy_process_group()


def distributed_output_path(path: str | Path) -> Path:
    """Return this rank's checkpoint path for distributed inference."""
    path = Path(path)
    if not dist.is_initialized():
        return path
    return path.with_name(f"{path.stem}.rank-{dist.get_rank()}-of-{dist.get_world_size()}{path.suffix}")
