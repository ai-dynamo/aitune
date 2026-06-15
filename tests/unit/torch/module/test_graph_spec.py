# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for graph spec."""

import pytest
import torch

from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_metadata import SampleMetadata


def _input_metadata(*shapes: tuple[int, ...], batch_size: int | None = None) -> SampleMetadata:
    return SampleMetadata.from_inputs(tuple(torch.randn(*shape) for shape in shapes), {}, batch_size=batch_size)


def _output_metadata(*shapes: tuple[int, ...], batch_size: int | None = None) -> SampleMetadata:
    outputs = tuple(torch.randn(*shape) for shape in shapes)
    return SampleMetadata.from_outputs(outputs[0] if len(outputs) == 1 else outputs, batch_size=batch_size)


def _graph_spec(batch_size: int | None = None) -> GraphSpec:
    return GraphSpec(
        name="test_graph",
        input_spec=_input_metadata((1, 3), (2, 5), batch_size=batch_size),
        output_spec=_output_metadata((1, 7), batch_size=batch_size),
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


def test_to_dict_and_from_dict_round_trip():
    graph_spec = _graph_spec(batch_size=1)
    graph_spec.update_shapes_seen(
        _input_metadata((4, 3), (8, 5), batch_size=4),
        _output_metadata((4, 7), batch_size=4),
    )

    restored = GraphSpec.from_dict(graph_spec.to_dict())

    assert restored == graph_spec


def test_from_dict_rejects_invalid_type():
    with pytest.raises(ValueError, match="Invalid dictionary format for GraphSpec"):
        GraphSpec.from_dict({"type": "TensorSpec"})


def test_string_representations_include_graph_name_and_specs():
    graph_spec = _graph_spec()

    assert str(graph_spec).startswith("Name=test_graph\nInput_spec:")
    assert "Output_spec:" in str(graph_spec)
    assert repr(graph_spec) == str(graph_spec)
