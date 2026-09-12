# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distributed lifecycle helpers for the LLM example."""

import os

import torch
import torch.distributed as dist


def initialize() -> bool:
    """Initialize NCCL when ``torchrun`` launches more than one process."""
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return False
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return True


def is_rank_zero() -> bool:
    """Return whether this process owns user-visible output."""
    return not dist.is_initialized() or dist.get_rank() == 0


def synchronize() -> None:
    """Wait for every rank to finish the current phase."""
    if dist.is_initialized():
        dist.barrier()


def shutdown() -> None:
    """Destroy the application-owned process group after normal or failed execution."""
    if dist.is_initialized():
        dist.destroy_process_group()
