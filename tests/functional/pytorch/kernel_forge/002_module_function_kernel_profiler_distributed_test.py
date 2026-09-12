# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate module function kernel profiling of a distributed model."""

# /// script
# docker_image = "nvcr.io/nvidia/pytorch:26.06-py3"
# scope = "always"
# allow_failure = false
# additional_tags = ["gpu/4"]
# ///

import logging
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.nn.parallel import DistributedDataParallel

from aitune.torch import ModuleFunctionKernelProfiler
from aitune.torch.distributed import coordinator

# Keep shared test utilities importable when this file runs as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.utilities.distributed_helpers import run_or_launch_distributed_test

WORLD_SIZE = 4
FEATURES = 8


class DistributedReluModel(nn.Module):
    """Small parameterized model with a directly observable functional call."""

    def __init__(self) -> None:
        """Initialize the model projection."""
        super().__init__()
        self.projection = nn.Linear(FEATURES, FEATURES)
        self.register_buffer("scale", torch.ones(()))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Run the projection followed by functional ReLU."""
        return F.relu(self.projection(value)) * self.scale


def _run_worker(device: torch.device) -> None:
    """Profile one DDP forward on each rank and validate its local result."""
    torch.manual_seed(dist.get_rank())
    local_rank = torch.cuda.current_device()
    model = DistributedDataParallel(DistributedReluModel().eval().to(device), device_ids=[local_rank])
    with torch.no_grad():
        model.module.scale.fill_(dist.get_rank() + 1)
    data = [((torch.randn(2, FEATURES, device=device),), {})]
    profiler = ModuleFunctionKernelProfiler(function_names={"relu"})

    profiling_df, function_data = profiler.profile(model, data, warmup_iterations=1)

    function_names = tuple(sorted(profiling_df["function_name"].unique()))
    coordinator.verify_equal(function_names, "profiled module functions")
    with coordinator.raise_if_any_rank_fails("Validating distributed module function profile"):
        assert "relu" in function_names
        assert len(function_data["relu"]) == 1
        assert model.module.scale.item() == 1.0
    coordinator.barrier()


def test_profile_distributed_module() -> None:
    """Run the parent launcher or the rank-local functional workflow."""
    run_or_launch_distributed_test(
        _run_worker,
        __file__,
        world_size=WORLD_SIZE,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    test_profile_distributed_module()
