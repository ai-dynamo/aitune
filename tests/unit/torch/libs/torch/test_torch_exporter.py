# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared Torch Export path."""

import pytest
import torch
import torch.nn as nn
from torch.fx.experimental.symbolic_shapes import ConstraintViolationError

from aitune.torch.libs.torch import TorchExporter
from tests.toy_models import TOY_EXPORT_MODELS
from tests.utilities.helpers import make_graph_spec, update_input_spec

SHAPE_CASES = (
    pytest.param([2], False, id="static"),
    pytest.param([2, 4], True, id="dynamic"),
)

TORCH_EXPORT_MODES = (
    pytest.param(True, True, id="inductor"),
    pytest.param(False, False, id="torch-tensorrt"),
)


def test_export_moves_nested_tensor_leaves_to_device(mocker):
    """Device preparation must include tensors nested below lists and dictionaries."""
    model = dict(TOY_EXPORT_MODELS)["complex"]().eval()
    sample = model.samples(batch_sizes=[2])[0]
    graph_spec = model.graph_spec(batch_sizes=[2])
    export = mocker.patch("torch.export.export")

    result = TorchExporter(strict=False).export(model, sample, graph_spec, device="meta")

    args = export.call_args.args[1]
    assert args[0].device.type == "meta"
    assert args[1][0].device.type == "meta"
    assert args[2]["residuals"][0].device.type == "meta"
    assert args[3].device.type == "meta"
    assert export.call_args.kwargs["kwargs"] is None
    assert result.sample[0] is args
    assert result.sample[1] == {}


def test_export_preserves_distributed_tensor_placement(mocker):
    """Device preparation must not move distributed tensor leaves."""
    model = dict(TOY_EXPORT_MODELS)["simple"]().eval()
    sample = model.samples(batch_sizes=[2])[0]
    graph_spec = model.graph_spec(batch_sizes=[2])
    tensor = sample[0][0]
    # Treat the sample tensor as a DTensor to verify that export preserves its application-managed placement instead
    # of moving it to the requested device.
    mocker.patch("aitune.torch.utils.module.DTensor", torch.Tensor)
    export = mocker.patch("torch.export.export")

    result = TorchExporter(strict=False).export(model, sample, graph_spec, device="meta")

    prepared_tensor = export.call_args.args[1][0]
    assert prepared_tensor.device == tensor.device
    assert prepared_tensor is not tensor
    assert result.sample[0][0] is prepared_tensor


@pytest.mark.parametrize(
    ("model_name", "model_factory"), TOY_EXPORT_MODELS, ids=[name for name, _ in TOY_EXPORT_MODELS]
)
@pytest.mark.parametrize(("batch_sizes", "is_dynamic"), SHAPE_CASES)
@pytest.mark.parametrize(("use_auto", "strict"), TORCH_EXPORT_MODES)
def test_export_model_matrix(use_auto, strict, model_name, model_factory, batch_sizes, is_dynamic):
    """Torch Export accepts each model structure and backend export policy."""
    del model_name
    model = model_factory().eval()
    samples = model.samples(batch_sizes=batch_sizes)
    graph_spec = model.graph_spec(batch_sizes=batch_sizes)

    export_result = TorchExporter(use_auto=use_auto, strict=strict).export(model, samples[0], graph_spec)

    assert (export_result.dynamic_shapes is not None) is is_dynamic
    exported_module = export_result.exported_program.module()
    for sample_args, sample_kwargs in samples:
        expected = model(*sample_args, **sample_kwargs)
        normalized = graph_spec.forward_signature.normalize(sample_args, sample_kwargs)
        actual = exported_module(*normalized.args, **normalized.kwargs)
        torch.testing.assert_close(actual, expected)


def test_export_retries_constraint_violations_with_bounded_dynamic_hints(mocker):
    """A constraint violation triggers a retry with bounded dynamic hints."""

    class StridedSpatialModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=7, stride=2, padding=3)
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        def forward(self, x):
            return self.pool(self.conv(x))

    model = StridedSpatialModel().eval()
    sample = ((torch.randn(1, 3, 128, 128),), {})
    graph_spec = make_graph_spec(model.forward, sample, batch_size=1)
    update_input_spec(graph_spec, ((torch.randn(1, 3, 192, 192),), {}), batch_size=1)
    torch_export = torch.export.export
    first_attempt = True

    def fail_first_attempt(*args, **kwargs):
        nonlocal first_attempt
        if first_attempt:
            first_attempt = False
            raise ConstraintViolationError("constraint violation")
        return torch_export(*args, **kwargs)

    export = mocker.patch("torch.export.export", side_effect=fail_first_attempt)

    result = TorchExporter(
        use_auto=False,
        strict=False,
        fallback_to_dynamic_hints=True,
    ).export(model, sample, graph_spec)

    assert export.call_count == 2
    height = result.dynamic_shapes["x"][2]
    width = result.dynamic_shapes["x"][3]
    assert isinstance(height, torch.export.dynamic_shapes._DimHint)
    assert (height.min, height.max) == (128, 192)
    assert height is width
    assert result.exported_program.module()(torch.randn(1, 3, 192, 192)).shape == (1, 4, 48, 48)


def test_export_does_not_retry_when_dynamic_shapes_cannot_be_relaxed(mocker):
    """A constraint failure without explicit dimensions must not repeat the same export."""
    model = nn.Linear(4, 2).eval()
    sample = ((torch.randn(2, 4),), {})
    graph_spec = make_graph_spec(model.forward, sample, batch_size=2)
    export = mocker.patch(
        "torch.export.export",
        side_effect=ConstraintViolationError("constraint violation"),
    )

    with pytest.raises(ConstraintViolationError, match="constraint violation"):
        TorchExporter(fallback_to_dynamic_hints=True).export(model, sample, graph_spec)

    export.assert_called_once()
