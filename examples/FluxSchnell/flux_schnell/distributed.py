# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distributed helpers owned by the Flux example."""

import os
from pathlib import Path

import torch
import torch.distributed as dist

from .config import Recipe


def initialize(recipe: Recipe) -> None:
    """Check the process count and initialize ranks launched by the user's torchrun command."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != recipe.gpu_count:
        raise RuntimeError(
            f"Recipe needs {recipe.gpu_count} ranks, but WORLD_SIZE is {world_size}. "
            f"Launch with torchrun --standalone --nproc-per-node={recipe.gpu_count} --module flux_schnell.<command> "
            f"--recipe {recipe.path}"
        )
    available = torch.cuda.device_count()
    if available < recipe.gpu_count:
        raise RuntimeError(f"Recipe needs {recipe.gpu_count} GPUs, but only {available} are visible")
    if world_size == 1:
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")


def is_rank_zero() -> bool:
    """Return whether this process owns user-visible output."""
    return not dist.is_initialized() or dist.get_rank() == 0


def synchronize() -> None:
    """Wait for every rank, or for local CUDA work outside distributed execution."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


def shutdown() -> None:
    """Destroy the application-owned process group after normal or failed execution."""
    if dist.is_initialized():
        dist.destroy_process_group()


def distributed_output_path(path: str | Path, rank: int | None = None, world_size: int | None = None) -> Path:
    """Return this rank's checkpoint path for distributed inference."""
    path = Path(path)
    world_size = world_size if world_size is not None else (dist.get_world_size() if dist.is_initialized() else 1)
    rank = rank if rank is not None else (dist.get_rank() if dist.is_initialized() else 0)
    if world_size == 1:
        return path
    return path.with_name(f"{path.stem}.rank-{rank}-of-{world_size}{path.suffix}")
