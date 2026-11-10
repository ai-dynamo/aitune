# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ONNX export through the flat tensor boundary."""

import onnx
import onnxruntime as ort
import pytest
import torch

from aitune.torch.dynamic_shapes import BatchDim, DynamicDim
from aitune.torch.libs.exporting import ONNXExporter
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.utils.tensor import format_tensor_name
from tests.toy_models import ToyNestedInputModel
from tests.utilities.helpers import make_graph_spec, make_recorded_graph_spec, update_input_spec


@pytest.mark.parametrize("use_dynamo", [False, True], ids=["trace", "dynamo"])
def test_export_runs_nested_input_and_output_model(tmp_path, use_dynamo):
    class NestedModule(torch.nn.Module):
        def forward(self, x, state):
            return {"result": x + state["bias"]}

    model = NestedModule().eval()
    x = torch.ones(2)
    bias = torch.full((2,), 3.0)
    sample = ((x, {"bias": bias}), {})
    graph_spec = make_graph_spec(model.forward, sample, model(*sample[0]))
    output_path = tmp_path / f"nested-{use_dynamo}.onnx"

    result = ONNXExporter(output_path, use_dynamo=use_dynamo).export(model, sample, graph_spec)

    exported = onnx.load(result)
    assert [value.name for value in exported.graph.input] == [
        format_tensor_name("x", "input"),
        format_tensor_name(("state", "bias"), "input"),
    ]
    assert [value.name for value in exported.graph.output] == [format_tensor_name("result", "output")]
    session = ort.InferenceSession(result.as_posix(), providers=["CPUExecutionProvider"])
    (actual,) = session.run(
        None,
        {
            format_tensor_name("x", "input"): x.numpy(),
            format_tensor_name(("state", "bias"), "input"): bias.numpy(),
        },
    )
    torch.testing.assert_close(torch.from_numpy(actual), x + bias)


@pytest.mark.skipif(
    tuple(int(component) for component in torch.__version__.split(".")[:2]) < (2, 10),
    reason="Requires PyTorch 2.10 or newer; PyTorch 2.9 generates invalid export guards referencing 'L'",
)
def test_dynamo_export_preserves_nested_dynamic_shapes(tmp_path):
    model = ToyNestedInputModel().eval()
    samples = model.samples(batch_sizes=[2, 4])
    graph_spec = model.graph_spec(batch_sizes=[2, 4])
    output_path = tmp_path / "nested-dynamic.onnx"

    ONNXExporter(output_path, use_dynamo=True).export(model, samples[0], graph_spec)

    exported = onnx.load(output_path)
    assert len(exported.graph.input) == len(graph_spec.input_spec.tensor_data)
    assert any(
        dimension.dim_param
        for value_info in exported.graph.input
        for dimension in value_info.type.tensor_type.shape.dim
    )


def test_dynamo_export_sorts_program_before_saving(mocker, tmp_path):
    """Keep the ONNX graph in dependency order before writing the artifact."""
    model = torch.nn.Identity().eval()
    x = torch.ones(2)
    sample = ((x,), {})
    graph_spec = make_graph_spec(model.forward, sample, model(x))
    output_path = tmp_path / "sorted.onnx"
    export = mocker.patch("torch.onnx.export")
    sort = mocker.patch("aitune.torch.libs.exporting.onnx_exporter.TopologicalSortPass")
    mocker.patch("onnx.checker.check_model")
    operations = mocker.Mock()
    operations.attach_mock(sort.return_value, "sort")
    operations.attach_mock(export.return_value.save, "save")

    ONNXExporter(output_path, use_dynamo=True).export(model, sample, graph_spec)

    assert operations.mock_calls == [
        mocker.call.sort(export.return_value.model),
        mocker.call.save(output_path, external_data=True),
    ]
    assert "f" not in export.call_args.kwargs


def test_trace_export_uses_explicit_dynamic_axis_names(tmp_path):
    class Add(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    model = Add().eval()
    sample = ((torch.ones(2, 4), torch.ones(2, 4)), {})
    graph_spec = make_graph_spec(model.forward, sample, model(*sample[0]))
    graph_spec.dynamic_shapes = {
        "x": (BatchDim("batch", min=1, opt=2, max=8), DynamicDim("sequence", min=1, opt=4, max=16)),
        "y": (BatchDim("batch", min=1, opt=2, max=8), DynamicDim("sequence", min=1, opt=4, max=16)),
    }
    output_path = tmp_path / "explicit-dynamic.onnx"

    ONNXExporter(output_path).export(model, sample, graph_spec)

    exported = onnx.load(output_path)
    dimensions = [dimension.dim_param for dimension in exported.graph.input[0].type.tensor_type.shape.dim]
    assert dimensions == ["batch", "sequence"]


def test_trace_export_marks_mutable_state_output_as_dynamic(tmp_path):
    class ReplaceCache(torch.nn.Module):
        def forward(self, x, cache):
            cache[0] = x + 1
            return x * 2

    model = ReplaceCache().eval()
    sample = ((torch.ones(2, 4), [None]), {})
    graph_spec = make_recorded_graph_spec(model, sample, strict=True)
    second_x = torch.ones(3, 4)
    second_cache = [None]
    model(second_x, second_cache)
    update_input_spec(graph_spec, ((second_x, [None]), {}), strict=True)
    graph_spec.post_input_spec.update_shapes_seen(
        SampleMetadata.from_inputs({"x": second_x, "cache": second_cache}, strict=True)
    )
    output_path = tmp_path / "dynamic-state.onnx"

    ONNXExporter(output_path).export(model, sample, graph_spec)

    exported = onnx.load(output_path)
    state_output = next(
        output for output in exported.graph.output if output.name == format_tensor_name(("cache", 0), "state")
    )
    assert state_output.type.tensor_type.shape.dim[0].dim_param


def test_failed_export_removes_incomplete_artifact(mocker, tmp_path):
    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    model = Identity().eval()
    x = torch.ones(2)
    sample = ((x,), {})
    graph_spec = make_graph_spec(model.forward, sample, model(x))
    output_path = tmp_path / "incomplete.onnx"

    def fail_after_creating_artifact(*args, **kwargs):
        del args
        output_path.write_bytes(b"incomplete")
        raise RuntimeError("export failed")

    mocker.patch("torch.onnx.export", side_effect=fail_after_creating_artifact)

    with pytest.raises(RuntimeError, match="export failed"):
        ONNXExporter(output_path).export(model, sample, graph_spec)

    assert not output_path.exists()
