# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic batch and spatial shape constraints across backends."""

from unittest.mock import patch

import pytest
import torch
from packaging.version import Version
from torch import nn

import aitune.torch as ait
from aitune.torch.dynamic_shapes import DynamicShapes
from aitune.torch.libs.exporting import torch_exporter
from tests.functional_ci.backends import BACKENDS, BackendCase, with_backends
from tests.functional_ci.lifecycle import TENSORRT_FP32, assert_matches_eager, pinned_strategy, selected_backend
from tests.functional_ci.models import IN_FEATURES, make_image_model, make_input, make_mlp, seeded

TORCH_2_8_TO_2_10 = Version(torch.__version__).release[:2] in ((2, 8), (2, 9), (2, 10))


@pytest.mark.matrix("aot_tuning.models.dynamic-batch-recorded")
@with_backends("base")
def test_recorded_batch_range(case: BackendCase, device: torch.device):
    """Batch bounds come from recorded batch sizes; seen and unseen in-range batch sizes match eager."""
    backend = case.factory()
    tuned = ait.Module(make_mlp(device), "mlp", strategy=pinned_strategy(backend))
    ait.tune(tuned, list(make_input(device, batch_size=8).unbind(0)), batch_sizes=[1, 4], device=device)

    (graph_spec,) = tuned.graph_specs
    assert (graph_spec.get_min_batch_size(), graph_spec.get_max_batch_size()) == (1, 4)
    selected_backend(tuned, type(backend))
    assert_matches_eager(tuned, make_mlp(device), device, case.tolerance, batch_sizes=(1, 2, 3, 4))


@pytest.mark.matrix("aot_tuning.models.dynamic-batch-user")
@with_backends("base")
def test_explicit_batch_dim(case: BackendCase, device: torch.device):
    """A user `BatchDim` makes the batch dimension dynamic beyond the single recorded batch size."""
    backend = case.factory()
    dynamic_shapes: DynamicShapes = {"input": (ait.BatchDim("batch", min=1, max=8, opt=4), IN_FEATURES)}
    tuned = ait.Module(make_mlp(device), "mlp", strategy=pinned_strategy(backend), dynamic_shapes=dynamic_shapes)
    ait.tune(tuned, list(make_input(device, batch_size=2).unbind(0)), batch_sizes=[2], device=device)

    (graph_spec,) = tuned.graph_specs
    assert graph_spec.dynamic_shapes == dynamic_shapes
    selected_backend(tuned, type(backend))
    assert_matches_eager(tuned, make_mlp(device), device, case.tolerance, batch_sizes=(1, 5, 8))


@pytest.mark.matrix("aot_tuning.models.dynamic-non-batch")
@with_backends(
    "base",
    marks={
        # AITune's `_activate` reloads the artifact via `torch_tensorrt.load`, which wraps the loader assertion in ValueError.
        "torch-tensorrt-aot": pytest.mark.xfail(
            TORCH_2_8_TO_2_10,
            reason="PyTorch 2.8–2.10's exported-program loader rejects symbolic 4D image strides (aitune#118)",
            raises=ValueError,
            strict=True,
        )
    },
)
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
    ait.tune(tuned, samples, batch_sizes=[1], device=device, ignore_failing_modules=False)

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

    ait.save(tuned, tmp_path / "model.ait")
    tuned.deactivate()
    # Other weights than the tuned model, so outputs match only if `ait.load` restores the checkpoint weights.
    restored = ait.load(make_image_model(device, seed=1), tmp_path / "model.ait")
    assert isinstance(restored, ait.Module)
    selected_backend(restored, type(backend))
    with torch.no_grad():
        for value in values:
            case.tolerance.assert_close(restored(value), reference(value))


@pytest.mark.matrix("aot_tuning.models.dynamic-non-batch/torch-tensorrt-aot")
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
