# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Functional coverage for graph breaks in model forward methods."""

import pytest
import torch
from torch import nn

import aitune.torch as ait
from tests.functional_ci.backends import BackendCase, with_backends
from tests.functional_ci.lifecycle import pinned_strategy, selected_backend
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
    """`graph_breaks` backends compile around the data-dependent branch; export backends fail to export it.

    Tuning sees only the positive branch, so the negative branch proves the split graph covers both.
    """
    backend = case.factory()
    tuned = ait.Module(BranchingModel(device), "branching", strategy=pinned_strategy(backend))
    value = make_input(device).abs()
    samples = list(value.unbind(0))
    if "graph_breaks" not in case.capabilities:
        with pytest.raises(Exception, match="(?i)data-dependent"):
            ait.tune(
                tuned,
                samples,
                batch_sizes=[value.shape[0]],
                device=device,
                ignore_failing_modules=False,
            )
        return

    ait.tune(tuned, samples, batch_sizes=[value.shape[0]], device=device)
    selected_backend(tuned, type(backend))
    reference = BranchingModel(device)
    with torch.no_grad():
        for branch_input in (value, -value):
            case.tolerance.assert_close(tuned(branch_input), reference(branch_input))
