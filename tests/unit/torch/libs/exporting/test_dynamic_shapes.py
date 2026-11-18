# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for flat-boundary dynamic-shape helpers."""

import pytest
import torch

from aitune.torch.dynamic_shapes import BatchDim, DynamicDim
from aitune.torch.libs.exporting.dynamic_shapes import build_dynamic_shapes, prepare_export_sample
from tests.utilities.helpers import make_graph_spec, update_input_spec


def nested_inputs(x, cache, options):
    return x, cache, options


def first_value(values):
    return values[0]


def single_input(x):
    return x


def two_inputs(x, y):
    return x, y


def flat_shapes(graph_spec, *, use_auto=True):
    dynamic_shapes = build_dynamic_shapes(graph_spec, use_auto=use_auto)
    assert dynamic_shapes is not None
    return dynamic_shapes["flat_inputs"]


def dynamic_graph_spec(first, second, *, first_batch_size=2, second_batch_size=2):
    sample = ((first,), {})
    graph_spec = make_graph_spec(single_input, sample, batch_size=first_batch_size)
    update_input_spec(graph_spec, ((second,), {}), batch_size=second_batch_size)
    return sample, graph_spec


def test_build_dynamic_shapes_returns_none_for_static_inputs():
    sample = ((torch.randn(2, 32),), {})
    graph_spec = make_graph_spec(single_input, sample, batch_size=2)

    assert build_dynamic_shapes(graph_spec) is None


@pytest.mark.parametrize(
    ("first", "second", "batch_sizes", "dynamic_axes"),
    [
        pytest.param((1, 32), (2, 32), (1, 2), {0}, id="batch"),
        pytest.param((2, 5), (2, 7), (2, 2), {1}, id="sequence"),
        pytest.param((1, 5), (2, 7), (1, 2), {0, 1}, id="batch-and-sequence"),
    ],
)
def test_build_dynamic_shapes_marks_varying_axes(first, second, batch_sizes, dynamic_axes):
    _, graph_spec = dynamic_graph_spec(
        torch.randn(*first),
        torch.randn(*second),
        first_batch_size=batch_sizes[0],
        second_batch_size=batch_sizes[1],
    )

    (axes,) = flat_shapes(graph_spec)

    assert axes.keys() == dynamic_axes


def test_build_dynamic_shapes_preserves_flat_tensor_order_for_nested_inputs():
    sample = (
        (
            torch.randn(2, 4),
            [torch.randn(2, 4), None],
            {"mask": torch.randn(2, 4), "scale": 1.0},
        ),
        {},
    )
    graph_spec = make_graph_spec(nested_inputs, sample, batch_size=2)
    update_input_spec(
        graph_spec,
        (
            (
                torch.randn(3, 4),
                [torch.randn(3, 4), None],
                {"mask": torch.randn(3, 4), "scale": 1.0},
            ),
            {},
        ),
        batch_size=3,
    )

    shapes = flat_shapes(graph_spec)

    assert len(shapes) == len(graph_spec.input_spec.tensor_data) == 3
    assert all(0 in axes for axes in shapes)


@pytest.mark.parametrize("use_auto", [True, False], ids=["automatic", "bounded"])
def test_build_dynamic_shapes_shares_matching_dimensions(use_auto):
    sample = ((torch.randn(2, 5), torch.randn(2, 5)), {})
    graph_spec = make_graph_spec(two_inputs, sample, batch_size=2)
    update_input_spec(graph_spec, ((torch.randn(2, 9), torch.randn(2, 9)), {}), batch_size=2)

    x_shapes, y_shapes = flat_shapes(graph_spec, use_auto=use_auto)

    assert x_shapes[1] is y_shapes[1]
    if not use_auto:
        assert (x_shapes[1].min, x_shapes[1].max) == (5, 9)


def test_build_dynamic_shapes_honors_explicit_definitions():
    sample = ((torch.randn(2, 4), torch.randn(2, 4)), {})
    graph_spec = make_graph_spec(two_inputs, sample, batch_size=2)
    graph_spec.dynamic_shapes = {
        "x": (BatchDim("batch", min=1, opt=2, max=8), DynamicDim("sequence", min=1, opt=4, max=16)),
        "y": (BatchDim("batch", min=1, opt=2, max=8), DynamicDim("sequence", min=1, opt=4, max=16)),
    }

    x_shapes, y_shapes = flat_shapes(graph_spec)
    explicit_x, explicit_y = flat_shapes(graph_spec, use_auto=False)

    assert x_shapes[0] is y_shapes[0]
    assert (x_shapes[0].min, x_shapes[0].max) == (1, 8)
    assert (x_shapes[1].min, x_shapes[1].max) == (1, 16)
    assert explicit_x[1] is explicit_y[1]
    assert explicit_x[1].__name__ == "sequence"


@pytest.mark.parametrize(
    ("first", "second", "batch_sizes", "expected_shape"),
    [
        pytest.param((2, 1), (2, 3), (2, 2), (2, 2), id="dynamic-axis"),
        pytest.param((1, 4), (2, 4), (1, 2), (2, 4), id="batch-axis"),
        pytest.param((2, 3), (2, 5), (2, 2), (2, 3), id="already-large"),
    ],
)
def test_prepare_export_sample_avoids_size_one_dynamic_hints(first, second, batch_sizes, expected_shape):
    sample, graph_spec = dynamic_graph_spec(
        torch.randn(*first),
        torch.randn(*second),
        first_batch_size=batch_sizes[0],
        second_batch_size=batch_sizes[1],
    )

    args, _ = prepare_export_sample(sample, graph_spec)

    assert args[0].shape == expected_shape


@pytest.mark.parametrize(
    "runtime_tail",
    [
        pytest.param([], id="missing-leaf"),
        pytest.param([None], id="changed-leaf-type"),
    ],
)
def test_prepare_export_sample_tolerates_runtime_structure_changes(runtime_tail):
    recorded = (([torch.ones(1), torch.ones(1)],), {})
    graph_spec = make_graph_spec(first_value, recorded, torch.ones(1))
    update_input_spec(graph_spec, (([torch.ones(3), torch.ones(3)],), {}))
    runtime_sample = (([torch.ones(1), *runtime_tail],), {})

    args, kwargs = prepare_export_sample(runtime_sample, graph_spec)

    assert args[0][0].shape == (2,)
    assert args[0][1:] == runtime_tail
    assert kwargs == {}


def test_prepare_export_sample_preserves_keyword_call_layout():
    first = torch.randn(1, 4)
    second = torch.randn(2, 4)
    sample = ((), {"x": first})
    graph_spec = make_graph_spec(single_input, sample, first, batch_size=1)
    update_input_spec(graph_spec, ((), {"x": second}), batch_size=2)

    args, kwargs = prepare_export_sample(sample, graph_spec)

    assert args == ()
    assert kwargs["x"].shape == (2, 4)
