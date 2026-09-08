# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ONNX export through the flat tensor boundary."""

import onnx
import onnxruntime as ort
import pytest
import torch

from aitune.torch.dynamic_shapes import BatchDim, DynamicDim
from aitune.torch.libs.exporting import ONNXExporter
from aitune.torch.utils.tensor import format_tensor_name
from tests.toy_models import ToyNestedInputModel
from tests.utilities.helpers import make_graph_spec


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
