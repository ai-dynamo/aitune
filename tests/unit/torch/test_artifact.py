# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Torch artifact tensor contracts."""

import pytest
import torch

from aitune.records import BoundedTensorSpec, DType
from aitune.torch.artifact import bounded_tensor_specs
from aitune.torch.dynamic_shapes import BatchDim, DynamicDim
from aitune.torch.module.forward_signature import ForwardSignature
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_metadata import SampleMetadata


def _graph_spec() -> GraphSpec:
    def forward(tokens, mask):
        return tokens, mask

    signature = ForwardSignature.from_callable(forward)
    graph_spec = None
    for batch_size in (1, 4):
        tokens = torch.zeros(batch_size, 8, dtype=torch.int64)
        mask = torch.zeros(batch_size * 2, 8, dtype=torch.bool)
        arguments = signature.normalize((tokens, mask), {}).arguments
        inputs = SampleMetadata.from_inputs(arguments, batch_size=batch_size)
        outputs = SampleMetadata.from_outputs((tokens, mask), batch_size=batch_size)
        if graph_spec is None:
            graph_spec = GraphSpec("graph", inputs, outputs, signature)
        else:
            graph_spec.update_shapes_seen(inputs, outputs)
    assert graph_spec is not None
    return graph_spec


def test_bounded_tensor_specs_derive_bounds_dtype_and_batch_axis_from_graph_spec():
    specs = bounded_tensor_specs(_graph_spec(), "input")

    assert specs == (
        BoundedTensorSpec(
            name="input_tokens",
            dtype=DType.INT64,
            min_shape=(1, 8),
            max_shape=(4, 8),
            batch_axis=0,
        ),
        BoundedTensorSpec(
            name="input_mask",
            dtype=DType.BOOL,
            min_shape=(2, 8),
            max_shape=(8, 8),
            batch_axis=None,
        ),
    )


def test_bounded_tensor_specs_apply_backend_order_and_names():
    specs = bounded_tensor_specs(
        _graph_spec(),
        "output",
        metadata_indices=(1, 0),
        artifact_names=("OUTPUT__0", "OUTPUT__1"),
    )

    assert tuple(spec.name for spec in specs) == ("OUTPUT__0", "OUTPUT__1")
    assert tuple(spec.dtype for spec in specs) == (DType.BOOL, DType.INT64)


@pytest.mark.parametrize("names", [("input_mask", "input_tokens"), ("input_mask",)])
def test_bounded_tensor_specs_select_recorded_names_in_executable_order(names):
    specs = bounded_tensor_specs(_graph_spec(), "input", recorded_names=names)

    assert tuple(spec.name for spec in specs) == names
    assert specs[0].dtype is DType.BOOL
    assert specs[0].max_shape == (8, 8)


def test_bounded_tensor_specs_reject_unknown_executable_names():
    with pytest.raises(ValueError, match="missing from the recorded graph"):
        bounded_tensor_specs(_graph_spec(), "input", recorded_names=("unknown",))


def test_output_bounds_follow_discovered_batch_size_recorded_in_graph_spec():
    graph_spec = _graph_spec()
    graph_spec.update_max_batch_size(8)

    specs = bounded_tensor_specs(graph_spec, "output")

    assert specs[0].min_shape == (1, 8)
    assert specs[0].max_shape == (8, 8)
    assert specs[0].batch_axis == 0
    assert specs[1].min_shape == (2, 8)
    assert specs[1].max_shape == (16, 8)
    assert specs[1].batch_axis is None
    assert graph_spec.output_spec.tensor_specs[0].max_shape == [8, 8]
    assert graph_spec.output_spec.tensor_specs[1].max_shape == [16, 8]


def test_output_bounds_intersect_effective_logical_input_batch_ranges():
    graph_spec = _graph_spec()
    locator, _ = graph_spec.input_spec.tensor_data[0]
    graph_spec.dynamic_shapes[locator.path] = (BatchDim("batch", min=2, max=8), 8)

    specs = bounded_tensor_specs(graph_spec, "output")

    # The second input still supports logical batches 1..4 (physical sizes 2..8).
    assert specs[0].min_shape == (2, 8)
    assert specs[0].max_shape == (4, 8)
    assert specs[1].min_shape == (4, 8)
    assert specs[1].max_shape == (8, 8)


def test_output_bounds_follow_explicit_batch_range_without_changing_other_dimensions():
    signature = ForwardSignature.from_callable(lambda tokens: tokens)
    graph_spec = None
    for batch_size in (1, 4):
        tokens = torch.zeros(batch_size, 8)
        inputs = SampleMetadata.from_inputs({"tokens": tokens}, batch_size=batch_size)
        outputs = SampleMetadata.from_outputs((tokens, torch.ones(5)), batch_size=batch_size)
        if graph_spec is None:
            graph_spec = GraphSpec("graph", inputs, outputs, signature)
        else:
            graph_spec.update_shapes_seen(inputs, outputs)
    assert graph_spec is not None
    locator, _ = graph_spec.input_spec.tensor_data[0]
    graph_spec.dynamic_shapes[locator.path] = (BatchDim("batch", min=2, max=8), 8)

    specs = bounded_tensor_specs(graph_spec, "output")

    assert specs[0].min_shape == (2, 8)
    assert specs[0].max_shape == (8, 8)
    assert specs[1].min_shape == (5,)
    assert specs[1].max_shape == (5,)
    assert specs[1].batch_axis is None


def test_output_bounds_reject_disjoint_input_batch_ranges():
    graph_spec = _graph_spec()
    locator, _ = graph_spec.input_spec.tensor_data[0]
    graph_spec.dynamic_shapes[locator.path] = (BatchDim("batch", min=8, max=16), 8)

    with pytest.raises(ValueError, match="no shared logical batch range"):
        bounded_tensor_specs(graph_spec, "output")


def test_output_bounds_preserve_observations_without_logical_input_batch_axes():
    graph_spec = _graph_spec()
    for index, (locator, _) in enumerate(graph_spec.input_spec.tensor_data):
        graph_spec.dynamic_shapes[locator.path] = (DynamicDim(f"length{index}", min=1, max=16), 8)

    specs = bounded_tensor_specs(graph_spec, "output")

    assert specs[0].max_shape == (4, 8)
    assert specs[1].max_shape == (8, 8)
