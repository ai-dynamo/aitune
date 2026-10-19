# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic shape and structured input/output cases."""

from collections.abc import Callable
from dataclasses import replace

import pytest
import torch
from torch import nn

import aitune.torch as ait
from tests.functional_ci.backends import BackendCase, with_backends
from tests.functional_ci.lifecycle import assert_matches_eager, pinned_strategy, selected_backend
from tests.functional_ci.models import IN_FEATURES, make_input, make_mlp


class StructuredModel(nn.Module):
    """Model with a nested dict input, non-tensor and optional arguments, and a nested dict/tuple output."""

    def __init__(self, mlp: nn.Module):
        """Wrap the toy MLP."""
        super().__init__()
        self.mlp = mlp

    def forward(self, x, extras: dict[str, torch.Tensor], scale: float = 1.0, mask: torch.Tensor | None = None):
        """Return the scaled MLP output and two row statistics."""
        hidden = self.mlp(x + extras["bias"]) * scale
        if mask is not None:
            hidden = hidden * mask
        return {"out": hidden, "stats": (hidden.sum(-1), hidden.amax(-1))}


@pytest.mark.matrix("models.dynamic-batch-recorded")
@with_backends("dynamic_batch")
def test_recorded_batch_range(case: BackendCase, device: torch.device):
    """Batch bounds come from recorded batch sizes; seen and unseen in-range batch sizes match eager."""
    backend = case.factory()
    tuned = ait.Module(make_mlp(device), "mlp", strategy=pinned_strategy(backend))
    ait.tune(tuned, list(make_input(device, batch_size=8).unbind(0)), batch_sizes=[1, 4], device=device)

    (graph_spec,) = tuned.graph_specs
    assert (graph_spec.get_min_batch_size(), graph_spec.get_max_batch_size()) == (1, 4)
    selected_backend(tuned, type(backend))
    assert_matches_eager(tuned, make_mlp(device), device, case.tolerance, batch_sizes=(1, 2, 3, 4))


@pytest.mark.matrix("models.dynamic-batch-user")
@with_backends("dynamic_batch")
def test_explicit_batch_dim(case: BackendCase, device: torch.device):
    """A user `BatchDim` widens the compiled range beyond the single recorded batch size."""
    backend = case.factory()
    dynamic_shapes = {"input": (ait.BatchDim("batch", min=1, max=8, opt=4), IN_FEATURES)}
    tuned = ait.Module(make_mlp(device), "mlp", strategy=pinned_strategy(backend), dynamic_shapes=dynamic_shapes)
    ait.tune(tuned, list(make_input(device, batch_size=2).unbind(0)), batch_sizes=[2], device=device)

    (graph_spec,) = tuned.graph_specs
    assert graph_spec.dynamic_shapes == dynamic_shapes
    selected_backend(tuned, type(backend))
    assert_matches_eager(tuned, make_mlp(device), device, case.tolerance, batch_sizes=(1, 5, 8))


@pytest.mark.matrix(
    "models.non-tensor-inputs",
    "models.nested-inputs",
    "models.structured-outputs",
)
@with_backends("structured_io")
def test_structured_io(case: BackendCase, device: torch.device):
    """Nested dict input, a float and a `None` argument, and a nested dict/tuple output survive tuning."""
    backend = case.factory()
    # Row sums in `stats` accumulate the per-element error over `OUT_FEATURES` values, so widen `atol`.
    tolerance = replace(case.tolerance, atol=5 * case.tolerance.atol)
    bias = torch.linspace(-1, 1, IN_FEATURES, device=device)
    reference = StructuredModel(make_mlp(device))
    tuned = ait.Module(StructuredModel(make_mlp(device)), "structured", strategy=pinned_strategy(backend))

    def pipeline(model: Callable) -> Callable:
        # Every tensor in the sample is batched, so the bias is repeated per row.
        return lambda x: model(x, {"bias": bias.repeat(x.shape[0], 1)}, scale=2.0, mask=None)

    ait.tune(pipeline(tuned), list(make_input(device, batch_size=8).unbind(0)), batch_sizes=[1, 4], device=device)

    selected_backend(tuned, type(backend))
    assert_matches_eager(pipeline(tuned), pipeline(reference), device, tolerance, batch_sizes=(1, 3, 4))
