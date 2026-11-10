# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Torch Export through the flat tensor boundary."""

import pytest
import torch

from aitune.torch.libs.exporting import TorchExporter
from tests.toy_models import ToyNestedInputModel
from tests.utilities.helpers import make_graph_spec, update_input_spec


@pytest.mark.parametrize("strict", [True, False], ids=["strict", "non-strict"])
def test_export_supports_both_tracing_modes(strict):
    model = torch.nn.Identity().eval()
    x = torch.ones(2)
    sample = ((x,), {})
    graph_spec = make_graph_spec(model.forward, sample, model(x))

    export_result = TorchExporter(strict=strict).export(model, sample, graph_spec)
    output = export_result.exported_program.module()(x)

    torch.testing.assert_close(output, (x,))


def test_export_falls_back_to_non_strict_mode(mocker):
    model = torch.nn.Identity().eval()
    x = torch.ones(2)
    sample = ((x,), {})
    graph_spec = make_graph_spec(model.forward, sample, model(x))
    real_export = torch.export.export

    def fail_strict_export(*args, **kwargs):
        if kwargs["strict"]:
            raise RuntimeError("strict export failed")
        return real_export(*args, **kwargs)

    export = mocker.patch("torch.export.export", side_effect=fail_strict_export)

    export_result = TorchExporter().export(model, sample, graph_spec)
    output = export_result.exported_program.module()(x)

    assert [call.kwargs["strict"] for call in export.call_args_list] == [True, False]
    torch.testing.assert_close(output, (x,))


@pytest.mark.parametrize("use_auto", [True, False], ids=["automatic", "bounded"])
def test_export_supports_nested_dynamic_inputs(use_auto):
    model = ToyNestedInputModel().eval()
    samples = model.samples(batch_sizes=[2, 4])
    graph_spec = model.graph_spec(batch_sizes=[2, 4])

    export_result = TorchExporter(use_auto=use_auto).export(model, samples[0], graph_spec)
    exported_module = export_result.exported_program.module(check_guards=False)

    assert export_result.dynamic_shapes is not None
    for args, kwargs in samples:
        with torch.no_grad():
            expected = model(*args, **kwargs)
        arguments = graph_spec.forward_signature.normalize(args, kwargs).arguments
        flat_inputs = graph_spec.input_spec.flatten_tensors(arguments)
        actual = exported_module(*flat_inputs)
        torch.testing.assert_close(actual, (expected,))


@pytest.mark.skipif(
    tuple(int(component) for component in torch.__version__.split(".")[:2]) < (2, 10),
    reason="Requires PyTorch 2.10 or newer; PyTorch 2.9 generates invalid export guards referencing 'L'",
)
def test_export_supports_different_batch_axis_multipliers():
    class Model(torch.nn.Module):
        def forward(self, x, doubled):
            return x + 1, doubled + 1

    model = Model().eval()
    samples = [
        ((torch.ones(1, 4), torch.ones(2, 4)), {}),
        ((torch.ones(2, 4), torch.ones(4, 4)), {}),
    ]
    graph_spec = make_graph_spec(model.forward, samples[0], model(*samples[0][0]), batch_size=1)
    update_input_spec(graph_spec, samples[1], batch_size=2)

    export_result = TorchExporter().export(model, samples[0], graph_spec)
    exported_module = export_result.exported_program.module()

    for args, _ in samples:
        expected = model(*args)
        actual = exported_module(*args)
        torch.testing.assert_close(actual, expected)
