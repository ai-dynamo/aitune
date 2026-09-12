# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate rank-zero kernel optimization plan selection across distributed workers.

The pytest parent relaunches this file with torchrun. Each worker's optimizer deliberately selects a different plan,
then the test verifies that every backend applies rank zero's plan.
"""

# /// script
# docker_image = "nvcr.io/nvidia/pytorch:26.06-py3"
# scope = "always"
# allow_failure = false
# additional_tags = ["gpu/4"]
# ///

import logging
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.nn.parallel import DistributedDataParallel

from aitune.torch.backend import KernelSelectorBackend, KernelSelectorBackendConfig, TorchEagerBackend
from aitune.torch.distributed import coordinator
from aitune.torch.kernel_forge.kernel_optimization_plan import KernelOptimizationPlan
from aitune.torch.kernel_forge.kernel_provider import KernelProvider
from aitune.torch.module.sample_store import SampleStore

# Keep shared test utilities importable when this file runs as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.utilities.distributed_helpers import run_or_launch_distributed_test

WORLD_SIZE = 4
FEATURES = 8
RANK_ZERO_INCREMENT = 1.0


class RankPlanProvider(KernelProvider):
    """Serializable ReLU provider carrying a rank-selected increment."""

    def __init__(self, increment: float) -> None:
        """Initialize the provider with its observable plan value."""
        super().__init__()
        self.increment = increment

    @property
    def supported_function(self) -> str:
        """Return the functional operation replaced by this provider."""
        return "relu"

    def _prepare(self, samples) -> bool:
        """Accept all samples for this deterministic test provider."""
        return True

    def _infer(self, value: torch.Tensor) -> torch.Tensor:
        """Apply ReLU and expose which rank's plan was installed."""
        return torch.relu(value) + self.increment

    def _to_dict(self):
        """Serialize the selected increment."""
        return {"increment": self.increment}

    @classmethod
    def _from_dict(cls, state_dict):
        """Restore the selected increment."""
        return cls(state_dict["increment"])


class RankPlanOptimizer:
    """Return a distinct prepared plan on each distributed rank."""

    def make_plan(self, function, data, *, module):
        """Build a plan whose increment identifies the local rank."""
        del function, data, module
        provider = RankPlanProvider(float(dist.get_rank() + 1))
        provider.prepare([])
        return KernelOptimizationPlan((provider,))


class ReluModel(nn.Module):
    """Small parameterized model suitable for DistributedDataParallel."""

    def __init__(self) -> None:
        """Initialize the projection used before the replaceable ReLU."""
        super().__init__()
        self.projection = nn.Linear(FEATURES, FEATURES)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Run the projection and replaceable functional ReLU."""
        return F.relu(self.projection(value))


def _run_worker(device: torch.device) -> None:
    """Build with rank-local plans and verify that rank zero's plan wins."""
    torch.manual_seed(0)
    local_rank = torch.cuda.current_device()
    model = DistributedDataParallel(ReluModel().to(device), device_ids=[local_rank])
    sample = torch.randn(2, FEATURES, device=device)
    with torch.no_grad():
        expected = model(sample) + RANK_ZERO_INCREMENT

    backend = KernelSelectorBackend(
        KernelSelectorBackendConfig(kernel_providers=RankPlanProvider(0.0)),
        TorchEagerBackend(),
    )
    backend._create_optimizer = RankPlanOptimizer

    with TemporaryDirectory(prefix=f"aitune-kernel-plan-rank-{dist.get_rank()}-") as directory:
        root = Path(directory)
        samples = SampleStore.from_samples([((sample,), {})], root, "samples")
        cache_dir = root / "backend"
        cache_dir.mkdir()
        backend.build(model, None, samples, device, cache_dir)

    runtime = backend._runtime
    assert runtime is not None
    selected_provider = runtime.plan.providers[0]
    assert isinstance(selected_provider, RankPlanProvider)
    selected_increment = selected_provider.increment
    actual = backend.infer(sample)
    coordinator.verify_equal(selected_increment, "kernel optimization plan")
    with coordinator.raise_if_any_rank_fails("Validating synchronized kernel optimization plan"):
        # all ranks should have the same plan i.e. increment value from rank zero
        assert selected_increment == RANK_ZERO_INCREMENT
        # after applied provider from the plan, value should match expected value
        torch.testing.assert_close(actual, expected)
    coordinator.barrier()


def test_distributed_kernel_optimizer_uses_rank_zero_plan() -> None:
    """Run the parent launcher or the rank-local functional workflow."""
    run_or_launch_distributed_test(
        _run_worker,
        __file__,
        world_size=WORLD_SIZE,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    test_distributed_kernel_optimizer_uses_rank_zero_plan()
