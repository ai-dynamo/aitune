# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Torch Export through the flat tensor boundary."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch.fx.experimental.symbolic_shapes import ConstraintViolationError

from aitune.torch.libs.exporting import TorchExportContractError, TorchExporter
from aitune.torch.libs.exporting.torch_exporter import _preflight_unlift
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


def test_preflight_rehomes_a_nested_lifted_tensor_constant(mocker):
    @dataclass(frozen=True)
    class InputSpec:
        kind: object
        target: str

    class Signature:
        def __init__(self, input_specs, output_specs):
            self.input_specs = input_specs
            self.output_specs = output_specs

    lifted = torch.ones(2)
    exported_program = mocker.Mock()
    exported_program.graph_module = torch.nn.Module()
    exported_program.constants = {"module.self_attn.lifted_tensor_0": lifted}
    exported_program.graph_signature = Signature(
        [InputSpec(SimpleNamespace(name="CONSTANT_TENSOR"), "module.self_attn.lifted_tensor_0")],
        [],
    )
    exported_program.module.side_effect = [AttributeError("self_attn"), mocker.sentinel.module]

    _preflight_unlift(exported_program)

    repaired = exported_program._graph_signature
    assert repaired.input_specs[0].target.startswith("_aitune_lifted_tensor_")
    assert "." not in repaired.input_specs[0].target
    assert getattr(exported_program.graph_module, repaired.input_specs[0].target) is lifted
    assert exported_program.constants[repaired.input_specs[0].target] is lifted
    assert "module.self_attn.lifted_tensor_0" not in exported_program.constants
    assert exported_program.module.call_count == 2


def test_preflight_reports_an_unrepairable_unlift_failure(mocker):
    exported_program = mocker.Mock()
    exported_program.graph_signature.input_specs = []
    exported_program.module.side_effect = AttributeError("missing state")

    with pytest.raises(TorchExportContractError, match="cannot be unlifted"):
        _preflight_unlift(exported_program)


def test_export_retries_rejected_constraints_with_bounded_dynamic_hints(mocker):
    class StridedSpatialModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, kernel_size=7, stride=2, padding=3)
            self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        def forward(self, x):
            return self.pool(self.conv(x))

    model = StridedSpatialModel().eval()
    sample = ((torch.randn(1, 3, 128, 128),), {})
    graph_spec = make_graph_spec(model.forward, sample, model(*sample[0]), batch_size=1)
    update_input_spec(graph_spec, ((torch.randn(1, 3, 192, 192),), {}), batch_size=1)
    real_export = torch.export.export
    first_attempt = True

    def reject_explicit_constraints(*args, **kwargs):
        nonlocal first_attempt
        if first_attempt:
            first_attempt = False
            raise ConstraintViolationError("constraint violation")
        return real_export(*args, **kwargs)

    export = mocker.patch("torch.export.export", side_effect=reject_explicit_constraints)

    result = TorchExporter(
        use_auto=False,
        strict=False,
        fallback_to_dynamic_hints=True,
    ).export(model, sample, graph_spec)

    assert export.call_count == 2
    assert result.dynamic_shapes is not None
    height = result.dynamic_shapes["flat_inputs"][0][2]
    width = result.dynamic_shapes["flat_inputs"][0][3]
    assert isinstance(height, torch.export.dynamic_shapes._DimHint)
    assert (height.min, height.max) == (128, 192)
    assert height is width
    # The bounded hints are asserted above; Torch 2.9's generated guards can reference an undefined `L`.
    output = result.exported_program.module(check_guards=False)(torch.randn(1, 3, 192, 192))
    assert output[0].shape == (1, 4, 48, 48)


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
