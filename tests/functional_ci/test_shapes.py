# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic shape and structured input/output cases."""

from collections.abc import Callable

import torch
from torch import nn

import aitune.torch as ait
from aitune.torch.backend import Backend
from tests.functional_ci.backends import BACKENDS, BackendCase, with_backends
from tests.functional_ci.lifecycle import Tolerance, selected_backend
from tests.functional_ci.models import IN_FEATURES, make_input, make_mlp

# Row sums accumulate the per-element TensorRT error over `OUT_FEATURES` values.
TENSORRT_ROW_SUM = Tolerance(rtol=1e-3, atol=5e-3)


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


def _strategy(backend: Backend) -> ait.OneBackendStrategy:
    """Select exactly `backend` without performance validation or max-batch discovery."""
    strategy = ait.OneBackendStrategy(backend).enable_performance_validation(False)
    strategy.enable_find_max_batch_size(False)
    return strategy


def _assert_matches_eager(
    tuned: Callable, reference: Callable, device: torch.device, batch_sizes, tolerance: Tolerance
):
    """Compare tuned and eager outputs for each batch size."""
    with torch.no_grad():
        for batch_size in batch_sizes:
            value = make_input(device, batch_size=batch_size, seed=10 + batch_size)
            torch.testing.assert_close(tuned(value), reference(value), rtol=tolerance.rtol, atol=tolerance.atol)


@with_backends("dynamic_batch")
def test_recorded_batch_range(case: BackendCase, device: torch.device):
    """Batch bounds come from recorded batch sizes; seen and unseen in-range batch sizes match eager."""
    tolerance = case.tolerance
    tuned = ait.Module(make_mlp(device), "mlp", strategy=_strategy(case.factory()))
    ait.tune(tuned, list(make_input(device, batch_size=8).unbind(0)), batch_sizes=[1, 4], device=device)

    (graph_spec,) = tuned.graph_specs
    assert (graph_spec.get_min_batch_size(), graph_spec.get_max_batch_size()) == (1, 4)
    assert type(selected_backend(tuned)) is type(case.factory())
    _assert_matches_eager(tuned, make_mlp(device), device, [1, 2, 3, 4], tolerance)


@with_backends("dynamic_batch")
def test_explicit_batch_dim(case: BackendCase, device: torch.device):
    """A user `BatchDim` widens the compiled range beyond the single recorded batch size."""
    tolerance = case.tolerance
    dynamic_shapes = {"input": (ait.BatchDim("batch", min=1, max=8, opt=4), IN_FEATURES)}
    tuned = ait.Module(make_mlp(device), "mlp", strategy=_strategy(case.factory()), dynamic_shapes=dynamic_shapes)
    ait.tune(tuned, list(make_input(device, batch_size=2).unbind(0)), batch_sizes=[2], device=device)

    (graph_spec,) = tuned.graph_specs
    assert graph_spec.dynamic_shapes == dynamic_shapes
    assert type(selected_backend(tuned)) is type(case.factory())
    _assert_matches_eager(tuned, make_mlp(device), device, [1, 5, 8], tolerance)


@with_backends("structured_io")
def test_structured_io(case: BackendCase, device: torch.device):
    """Nested dict input, a float and a `None` argument, and a nested dict/tuple output survive tuning."""
    tolerance = TENSORRT_ROW_SUM if case is BACKENDS["tensorrt"] else case.tolerance
    bias = torch.linspace(-1, 1, IN_FEATURES, device=device)
    reference = StructuredModel(make_mlp(device))
    tuned = ait.Module(StructuredModel(make_mlp(device)), "structured", strategy=_strategy(case.factory()))

    def pipeline(model: Callable) -> Callable:
        # Every tensor in the sample is batched, so the bias is repeated per row.
        return lambda x: model(x, {"bias": bias.repeat(x.shape[0], 1)}, scale=2.0, mask=None)

    ait.tune(pipeline(tuned), list(make_input(device, batch_size=8).unbind(0)), batch_sizes=[1, 4], device=device)

    assert type(selected_backend(tuned)) is type(case.factory())
    _assert_matches_eager(pipeline(tuned), pipeline(reference), device, [1, 3, 4], tolerance)
