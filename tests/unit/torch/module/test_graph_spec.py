# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for graph spec."""

import pytest
import torch

from aitune.torch.dynamic_shapes import BatchDim, DynamicDim
from aitune.torch.module.forward_signature import ForwardSignature
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_metadata import SampleMetadata
from tests.utilities.helpers import make_input_metadata


def _forward(x, y):
    return x, y


FORWARD_SIGNATURE = ForwardSignature.from_callable(_forward)


def _input_metadata(*shapes: tuple[int, ...], batch_size: int | None = None) -> SampleMetadata:
    inputs = {name: torch.randn(*shape) for name, shape in zip(("x", "y"), shapes, strict=True)}
    return SampleMetadata.from_inputs(inputs, batch_size=batch_size)


def _output_metadata(*shapes: tuple[int, ...], batch_size: int | None = None) -> SampleMetadata:
    outputs = tuple(torch.randn(*shape) for shape in shapes)
    return SampleMetadata.from_outputs(outputs[0] if len(outputs) == 1 else outputs, batch_size=batch_size)


def _graph_spec(batch_size: int | None = None) -> GraphSpec:
    return GraphSpec(
        name="test_graph",
        input_spec=_input_metadata((1, 3), (2, 5), batch_size=batch_size),
        output_spec=_output_metadata((1, 7), batch_size=batch_size),
        forward_signature=FORWARD_SIGNATURE,
    )


def _batching_graph_spec() -> GraphSpec:
    def forward(x, *, mask):
        return x, mask

    forward_signature = ForwardSignature.from_callable(forward)
    input_spec = make_input_metadata(
        forward_signature,
        ((torch.randn(2, 3),), {"mask": torch.randn(4, 5)}),
        batch_size=2,
    )
    input_spec.update_shapes_seen(
        make_input_metadata(
            forward_signature,
            ((torch.randn(3, 3),), {"mask": torch.randn(6, 5)}),
            batch_size=3,
        )
    )
    return GraphSpec(
        name="batching_graph",
        input_spec=input_spec,
        output_spec=_output_metadata((2, 7), batch_size=2),
        forward_signature=forward_signature,
    )


def test_update_shapes_seen_updates_input_and_output_specs():
    graph_spec = _graph_spec(batch_size=1)
    inputs_metadata = _input_metadata((4, 3), (8, 5), batch_size=4)
    outputs_metadata = _output_metadata((4, 7), batch_size=4)

    graph_spec.update_shapes_seen(inputs_metadata, outputs_metadata)

    assert graph_spec.input_spec.tensor_specs[0].shape == ["batch0", 3]
    assert graph_spec.input_spec.tensor_specs[0].min_shape == [1, 3]
    assert graph_spec.input_spec.tensor_specs[0].max_shape == [4, 3]
    assert graph_spec.input_spec.tensor_specs[1].shape == ["batch0", 5]
    assert graph_spec.input_spec.tensor_specs[1].min_shape == [2, 5]
    assert graph_spec.input_spec.tensor_specs[1].max_shape == [8, 5]
    assert graph_spec.output_spec.tensor_specs[0].shape == ["batch0", 7]
    assert graph_spec.output_spec.tensor_specs[0].max_shape == [4, 7]


def test_make_batch_normalizes_call_and_resizes_inputs():
    graph_spec = _batching_graph_spec()

    args, kwargs = graph_spec.make_batch(
        (),
        {"mask": torch.randn(2, 5), "x": torch.randn(1, 3)},
        batch_size=5,
    )

    assert args[0].shape == (5, 3)
    assert kwargs["mask"].shape == (10, 5)


def test_update_max_batch_size_normalizes_call_and_updates_input_spec():
    graph_spec = _batching_graph_spec()

    graph_spec.update_max_batch_size(
        ((), {"mask": torch.randn(2, 5), "x": torch.randn(1, 3)}),
        max_batch_size=8,
    )

    tensor_specs = {locator.path: tensor_spec for locator, tensor_spec in graph_spec.input_spec.tensor_data}
    assert tensor_specs["x"].max_shape == [8, 3]
    assert tensor_specs["mask"].max_shape == [16, 5]


def test_get_max_batch_size_returns_largest_local_batch_axis():
    graph_spec = _graph_spec(batch_size=1)
    graph_spec.update_shapes_seen(
        _input_metadata((4, 3), (8, 5), batch_size=4),
        _output_metadata((4, 7), batch_size=4),
    )

    assert graph_spec.get_max_batch_size() == 8


def test_get_max_batch_size_normalized_divides_batch_axes_by_multipliers():
    graph_spec = _graph_spec(batch_size=1)
    graph_spec.update_shapes_seen(
        _input_metadata((4, 3), (8, 5), batch_size=4),
        _output_metadata((4, 7), batch_size=4),
    )

    assert graph_spec.get_max_batch_size(normalized=True) == 4


def test_get_max_batch_size_returns_one_without_batch_axes():
    graph_spec = _graph_spec()

    assert graph_spec.get_max_batch_size() == 1
    assert graph_spec.get_max_batch_size(normalized=True) == 1


def test_get_min_batch_size_returns_smallest_local_batch_axis():
    graph_spec = _graph_spec(batch_size=1)
    graph_spec.update_shapes_seen(
        _input_metadata((4, 3), (8, 5), batch_size=4),
        _output_metadata((4, 7), batch_size=4),
    )

    assert graph_spec.get_min_batch_size() == 1


def test_get_min_batch_size_returns_none_without_batch_axes():
    graph_spec = _graph_spec()

    assert graph_spec.get_min_batch_size() is None


def test_explicit_shapes_override_inferred_ranges():
    graph_spec = _graph_spec(batch_size=1)
    graph_spec.dynamic_shapes = {
        "x": (BatchDim("batch", min=1, opt=2, max=4), 3),
        "y": (2, DynamicDim("sequence", min=1, opt=8, max=16)),
    }
    tensor_data = {locator.path: (locator, tensor_spec) for locator, tensor_spec in graph_spec.input_spec.tensor_data}

    assert graph_spec.get_effective_input_shapes(*tensor_data["x"]) == ([1, 3], [2, 3], [4, 3])
    assert graph_spec.get_effective_input_shapes(*tensor_data["y"]) == ([2, 1], [2, 8], [2, 16])
    assert graph_spec.get_min_batch_size() == 1
    assert graph_spec.get_max_batch_size() == 4


def test_to_dict_and_from_dict_round_trip():
    graph_spec = _graph_spec(batch_size=1)
    graph_spec.update_shapes_seen(
        _input_metadata((4, 3), (8, 5), batch_size=4),
        _output_metadata((4, 7), batch_size=4),
    )

    restored = GraphSpec.from_dict(graph_spec.to_dict())

    assert restored == graph_spec


def test_explicit_shapes_round_trip(tmp_path):
    graph_spec = _graph_spec(batch_size=1)
    graph_spec.dynamic_shapes = {
        "x": (BatchDim("batch", min=1, opt=2, max=4), 3),
        ("options", "mask"): (2, DynamicDim("sequence", min=1, opt=8, max=16)),
    }

    state = graph_spec.to_dict()
    checkpoint = tmp_path / "graph_spec.pt"
    torch.save(state, checkpoint)
    restored = GraphSpec.from_dict(torch.load(checkpoint, weights_only=False))

    assert state["dynamic_shapes"] == graph_spec.dynamic_shapes
    assert restored == graph_spec


def test_from_dict_rejects_invalid_type():
    with pytest.raises(ValueError, match="Invalid dictionary format for GraphSpec"):
        GraphSpec.from_dict({"type": "TensorSpec"})


def test_string_representations_include_graph_name_and_specs():
    graph_spec = _graph_spec()

    assert str(graph_spec).startswith("Name=test_graph\nInput_spec:")
    assert "Output_spec:" in str(graph_spec)
    assert repr(graph_spec) == str(graph_spec)
