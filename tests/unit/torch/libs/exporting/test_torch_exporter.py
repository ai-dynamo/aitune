# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Torch Export through the flat tensor boundary."""

import inspect
import socket
from functools import partial

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module
from torch.fx.experimental.symbolic_shapes import ConstraintViolationError
from torch.utils import _pytree

from aitune.torch.libs.exporting import TorchExporter
from tests.toy_models import ToyNestedInputModel
from tests.utilities.helpers import make_graph_spec, update_input_spec

# PyTorch 2.8 lacks check_guards; inspect the API because CI also uses prerelease builds.
_MODULE_GUARD_KWARGS = (
    {"check_guards": False}
    if "check_guards" in inspect.signature(torch.export.ExportedProgram.module).parameters
    else {}
)


class BatchMultiplierModel(torch.nn.Module):
    def forward(self, x, doubled):
        return x + 1, doubled + 1


def fail_strict_export(real_export, *args, **kwargs):
    if kwargs["strict"]:
        raise RuntimeError("strict export failed")
    return real_export(*args, **kwargs)


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

    export = mocker.patch("torch.export.export", side_effect=partial(fail_strict_export, real_export))

    export_result = TorchExporter().export(model, sample, graph_spec)
    output = export_result.exported_program.module()(x)

    assert [call.kwargs["strict"] for call in export.call_args_list] == [True, False]
    torch.testing.assert_close(output, (x,))


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
    output = result.exported_program.module(**_MODULE_GUARD_KWARGS)(torch.randn(1, 3, 192, 192))
    assert output[0].shape == (1, 4, 48, 48)


@pytest.mark.parametrize("use_auto", [True, False], ids=["automatic", "bounded"])
def test_export_supports_nested_dynamic_inputs(use_auto):
    model = ToyNestedInputModel().eval()
    samples = model.samples(batch_sizes=[2, 4])
    graph_spec = model.graph_spec(batch_sizes=[2, 4])

    export_result = TorchExporter(use_auto=use_auto).export(model, samples[0], graph_spec)
    exported_module = export_result.exported_program.module(**_MODULE_GUARD_KWARGS)

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
    model = BatchMultiplierModel().eval()
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


@pytest.fixture
def dtensor_spec_registry():
    """Remove the process-wide DTensorSpec registration added by an export."""
    registered = DTensorSpec in _pytree.SUPPORTED_NODES
    yield
    if not registered and DTensorSpec in _pytree.SUPPORTED_NODES:
        _pytree._deregister_pytree_node(DTensorSpec)


@pytest.fixture
def single_rank_mesh(tmp_path, monkeypatch):
    interfaces = {name for _, name in socket.if_nameindex()}
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo" if "lo" in interfaces else "lo0")
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("strict", [True, False], ids=["strict", "non-strict"])
def test_export_supports_tensor_parallel_modules(strict, single_rank_mesh, dtensor_spec_registry):
    model = parallelize_module(torch.nn.Linear(3, 4), single_rank_mesh, ColwiseParallel()).eval()
    x = torch.randn(2, 3)
    sample = ((x,), {})
    expected = model(x)
    graph_spec = make_graph_spec(model.forward, sample, expected)

    export_result = TorchExporter(strict=strict).export(model, sample, graph_spec)
    output = export_result.exported_program.module()(x)

    torch.testing.assert_close(output, (expected,))


def test_export_does_not_register_dtensor_spec_for_dynamic_shapes(mocker, single_rank_mesh, dtensor_spec_registry):
    if DTensorSpec in _pytree.SUPPORTED_NODES:
        pytest.skip("DTensorSpec is already registered in this process")
    model = parallelize_module(torch.nn.Linear(3, 4), single_rank_mesh, ColwiseParallel()).eval()
    samples = [((torch.randn(2, 3),), {}), ((torch.randn(4, 3),), {})]
    graph_spec = make_graph_spec(model.forward, samples[0], model(*samples[0][0]), batch_size=2)
    update_input_spec(graph_spec, samples[1], batch_size=4)
    mocker.patch("torch.export.export")

    TorchExporter().export(model, samples[0], graph_spec)

    assert DTensorSpec not in _pytree.SUPPORTED_NODES


def test_export_does_not_register_dtensor_spec_for_local_modules(dtensor_spec_registry):
    if DTensorSpec in _pytree.SUPPORTED_NODES:
        pytest.skip("DTensorSpec is already registered in this process")
    model = torch.nn.Identity().eval()
    x = torch.ones(2)
    sample = ((x,), {})
    graph_spec = make_graph_spec(model.forward, sample, model(x))

    TorchExporter().export(model, sample, graph_spec)

    assert DTensorSpec not in _pytree.SUPPORTED_NODES
