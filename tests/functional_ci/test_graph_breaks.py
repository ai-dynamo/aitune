# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Functional coverage for graph breaks in model forward methods."""

import pytest
import torch
from torch import nn

import aitune.torch as ait
from aitune.torch.backend import TorchEagerBackend
from tests.functional_ci.backends import BackendCase, with_backends
from tests.functional_ci.lifecycle import selected_backend
from tests.functional_ci.models import make_input, make_mlp


class BranchingModel(nn.Module):
    """Run one of two seeded MLPs, picked by the sign of the batch mean."""

    def __init__(self, device: torch.device):
        """Build the two seeded branches on `device`."""
        super().__init__()
        self.positive = make_mlp(device)
        self.negative = make_mlp(device, seed=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the branch selected by the batch mean."""
        # Data-dependent branch: torch.compile breaks the graph here and torch.export rejects it.
        if x.mean() > 0:
            return self.positive(x)
        return self.negative(x)


@pytest.mark.matrix("models.graph-breaks")
@with_backends()
def test_graph_break(case: BackendCase, device: torch.device):
    """Tune on the positive branch only; the negative branch must still match eager.

    `graph_breaks` backends compile around the break; export backends fail to export, so the strategy falls back to
    Torch eager instead of a graph traced through one branch.
    """
    backend = case.factory()
    strategy = ait.FirstWinsStrategy([backend, TorchEagerBackend()])
    strategy.enable_performance_validation(False)
    strategy.enable_find_max_batch_size(False)
    tuned = ait.Module(BranchingModel(device), "branching", strategy=strategy)
    value = make_input(device).abs()
    ait.tune(tuned, list(value.unbind(0)), batch_sizes=[value.shape[0]], device=device)

    selected_backend(tuned, type(backend) if "graph_breaks" in case.capabilities else TorchEagerBackend)
    reference = BranchingModel(device)
    with torch.no_grad():
        for branch_input in (value, -value):
            case.tolerance.assert_close(tuned(branch_input), reference(branch_input))
