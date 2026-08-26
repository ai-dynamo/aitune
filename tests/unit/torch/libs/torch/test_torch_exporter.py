# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared Torch Export path."""

import pytest
import torch

from aitune.torch.libs.torch import TorchExporter
from tests.toy_models import TOY_EXPORT_MODELS

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
