# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic shape and structured input/output cases."""

from collections.abc import Callable
from dataclasses import replace
from typing import cast
from unittest.mock import patch

import pytest
import torch
from packaging.version import Version
from torch import nn

import aitune.torch as ait
from aitune.torch.dynamic_shapes import DynamicShapes
from aitune.torch.libs.torch import torch_exporter
from tests.functional_ci.backends import BACKENDS, BackendCase, with_backends
from tests.functional_ci.lifecycle import TENSORRT_FP32, assert_matches_eager, pinned_strategy, selected_backend
from tests.functional_ci.models import IN_FEATURES, make_image_model, make_input, make_mlp, seeded


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


class NestedInputModel(nn.Module):
    """Model with only tensor inputs, shaped like the Flux transformer inputs in legacy test 044."""

    def __init__(self, mlp: nn.Module):
        """Wrap the toy MLP."""
        super().__init__()
        self.mlp = mlp

    def forward(
        self,
        x,
        extras: dict[str, torch.Tensor],
        residuals: list[torch.Tensor],
        positions: torch.Tensor,
        timestep: torch.Tensor,
    ):
        """Return the MLP output for the shifted input, scaled per row by `timestep`."""
        hidden = x + extras["bias"] + residuals[0] - residuals[1] + positions
        return self.mlp(hidden * timestep[:, None])


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
    dynamic_shapes: DynamicShapes = {"input": (ait.BatchDim("batch", min=1, max=8, opt=4), IN_FEATURES)}
    tuned = ait.Module(make_mlp(device), "mlp", strategy=pinned_strategy(backend), dynamic_shapes=dynamic_shapes)
    ait.tune(tuned, list(make_input(device, batch_size=2).unbind(0)), batch_sizes=[2], device=device)

    (graph_spec,) = tuned.graph_specs
    assert graph_spec.dynamic_shapes == dynamic_shapes
    selected_backend(tuned, type(backend))
    assert_matches_eager(tuned, make_mlp(device), device, case.tolerance, batch_sizes=(1, 5, 8))


@pytest.mark.matrix("models.dynamic-non-batch")
@with_backends("dynamic_non_batch")
def test_explicit_spatial_dims(case: BackendCase, device: torch.device, tmp_path):
    """Tune from one 24x24 sample with a user `BatchDim` and spatial `DynamicDim`, then run other batch and image sizes.

    The tuned module and the module reloaded from its checkpoint both match eager at sizes that were not recorded.
    """
    backend = case.factory()
    spatial = ait.DynamicDim("spatial", min=16, opt=24, max=32)
    dynamic_shapes: DynamicShapes = {"input": (ait.BatchDim("batch", min=1, opt=2, max=2), 3, spatial, spatial)}
    tuned = ait.Module(
        make_image_model(device), "image", strategy=pinned_strategy(backend), dynamic_shapes=dynamic_shapes
    )
    generator = torch.Generator(device=device).manual_seed(1)
    samples = list(torch.randn(1, 3, 24, 24, device=device, generator=generator).unbind(0))
    if case is BACKENDS["torch-tensorrt-aot"] and Version(torch.__version__).release[:2] in ((2, 8), (2, 9), (2, 10)):
        # AITune's `_activate` reloads the artifact via `torch_tensorrt.load`, which wraps the assertion in ValueError.
        with pytest.raises(ValueError, match="valid Torchscript module or ExportedProgram"):
            ait.tune(tuned, samples, batch_sizes=[1], device=device, ignore_failing_modules=False)
        pytest.xfail("PyTorch 2.8–2.10's exported-program loader rejects symbolic 4D image strides (aitune#118)")
    ait.tune(tuned, samples, batch_sizes=[1], device=device)

    (graph_spec,) = tuned.graph_specs
    assert graph_spec.dynamic_shapes == dynamic_shapes
    selected_backend(tuned, type(backend))
    reference = make_image_model(device)
    values = [
        torch.randn(batch_size, 3, size, size, device=device, generator=generator)
        for batch_size, size in ((1, 16), (2, 24), (2, 32), (1, 23))
    ]
    with torch.no_grad():
        for value in values:
            case.tolerance.assert_close(tuned(value), reference(value))

    ait.save(cast(nn.Module, tuned), tmp_path / "model.ait")
    tuned.deactivate()
    # Other weights than the tuned model, so outputs match only if `ait.load` restores the checkpoint weights.
    restored = ait.load(make_image_model(device, seed=1), tmp_path / "model.ait")
    assert isinstance(restored, ait.Module)
    selected_backend(restored, type(backend))
    with torch.no_grad():
        for value in values:
            case.tolerance.assert_close(restored(value), reference(value))


@pytest.mark.matrix("models.dynamic-non-batch/torch-tensorrt-aot")
def test_spatial_dims_export_retry(device: torch.device):
    """Torch-TensorRT AOT exports again with bounded dynamic hints if strict export rejects the user's spatial dims.

    `PixelUnshuffle(2)` needs even H and W, but the `DynamicDim` allows odd sizes, so strict export fails once.
    """
    with seeded(device, 0):
        model = nn.Sequential(
            nn.Unflatten(1, (1, -1)),
            nn.PixelUnshuffle(2),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )
        model = model.to(device).eval()
        values = [torch.randn(1, size, size, device=device) for size in (16, 24, 32, 22)]
    with torch.no_grad():
        expected = [model(value) for value in values]

    backend = BACKENDS["torch-tensorrt-aot"].factory()
    spatial = ait.DynamicDim("spatial", min=16, opt=24, max=32)
    tuned = ait.Module(
        model,
        "even-spatial",
        strategy=pinned_strategy(backend),
        dynamic_shapes={"input": (1, spatial, spatial)},
    )
    with patch.object(torch_exporter, "_as_dynamic_hints", wraps=torch_exporter._as_dynamic_hints) as retry:
        ait.tune(tuned, list(values[1].unbind(0)), batch_sizes=[1], device=device)
    retry.assert_called_once()
    selected_backend(tuned, type(backend))
    with torch.no_grad():
        for value, output in zip(values, expected, strict=True):
            TENSORRT_FP32.assert_close(tuned(value), output)


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


@pytest.mark.matrix("models.nested-inputs/torch-tensorrt-aot")
@pytest.mark.parametrize("batch_sizes", [[2], [2, 4]], ids=["static", "dynamic"])
def test_nested_tensor_inputs(batch_sizes: list[int], device: torch.device):
    """Torch-TensorRT AOT tunes nested tensor inputs passed as kwargs at static and dynamic batch, as in legacy 044.

    The inputs are a dict, a list, an unbatched tensor, and a 1-D batched scalar. `test_structured_io` also passes
    non-tensor arguments, which this backend cannot save yet, so it does not run on this backend.
    """
    case = BACKENDS["torch-tensorrt-aot"]
    backend = case.factory()
    bias = torch.linspace(-1, 1, IN_FEATURES, device=device)
    residuals = [bias.flip(0), 0.5 * bias]
    reference = NestedInputModel(make_mlp(device))
    tuned = ait.Module(NestedInputModel(make_mlp(device)), "nested", strategy=pinned_strategy(backend))

    def pipeline(model: Callable) -> Callable:
        # `positions` keeps one shape at every batch size, like `img_ids` in 044; the other tensors are batched.
        return lambda x: model(
            x,
            extras={"bias": bias.repeat(x.shape[0], 1)},
            residuals=[residual.repeat(x.shape[0], 1) for residual in residuals],
            positions=bias.flip(0),
            timestep=torch.linspace(0.5, 1.5, x.shape[0], device=device),
        )

    ait.tune(pipeline(tuned), list(make_input(device, batch_size=8).unbind(0)), batch_sizes=batch_sizes, device=device)

    selected_backend(tuned, type(backend))
    check_sizes = range(batch_sizes[0], batch_sizes[-1] + 1)
    assert_matches_eager(pipeline(tuned), pipeline(reference), device, case.tolerance, batch_sizes=check_sizes)
