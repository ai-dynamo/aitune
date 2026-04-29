# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aitune.torch.utils.shapes."""

from unittest.mock import Mock

import pytest

from aitune.torch.module.locator import Locator, ObjectType
from aitune.torch.utils.shapes import (
    _create_nested_structure,
    _raise_on_locator_user_type,
    create_inputs_mapping,
    create_ordered_dynamic_shapes,
    war_for_positional_arguments,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(name, min_shape, max_shape):
    """Create a minimal mock TensorSpec."""
    ts = Mock()
    ts.name = name
    ts.min_shape = list(min_shape)
    ts.max_shape = list(max_shape)
    return ts


def _mock_input_spec(tensor_data):
    spec = Mock()
    spec.tensor_data = tensor_data
    return spec


def _locator(leaf_name, depth=1):
    loc = Mock()
    loc.leaf_name = leaf_name
    loc.depth = depth
    return loc


# ---------------------------------------------------------------------------
# _create_nested_structure
# ---------------------------------------------------------------------------


def test_create_nested_structure_dict_list():
    locator = Locator((("zw", ObjectType.DICT), (0, ObjectType.SEQUENCE)))
    assert _create_nested_structure(locator, last={}) == {"zw": [{}]}


def test_create_nested_structure_three_levels():
    locator = Locator((("zw", ObjectType.DICT), (0, ObjectType.SEQUENCE), ("t", ObjectType.DICT)))
    assert _create_nested_structure(locator, last={}) == {"zw": [{"t": {}}]}


def test_create_nested_structure_extend_existing_list():
    locator = Locator((("zw", ObjectType.DICT), (0, ObjectType.SEQUENCE)))
    root = _create_nested_structure(locator, last={})
    locator2 = Locator((("zw", ObjectType.DICT), (1, ObjectType.SEQUENCE)))
    root = _create_nested_structure(locator2, root=root, last={})
    assert root == {"zw": [{}, {}]}


def test_create_nested_structure_scalar_leaf():
    locator = Locator((("zw", ObjectType.DICT),))
    assert _create_nested_structure(locator, last=1) == {"zw": 1}


# ---------------------------------------------------------------------------
# _raise_on_locator_user_type
# ---------------------------------------------------------------------------


def test_raise_on_locator_user_type_raises_for_user_type():
    locator = Locator((("x", ObjectType.USER_TYPE),))
    with pytest.raises(ValueError, match="user types"):
        _raise_on_locator_user_type(locator)


def test_raise_on_locator_user_type_raises_for_dataclass():
    locator = Locator((("x", ObjectType.DATACLASS),))
    with pytest.raises(ValueError):
        _raise_on_locator_user_type(locator)


def test_raise_on_locator_user_type_ok_for_dict_and_sequence():
    locator = Locator((("x", ObjectType.DICT), (0, ObjectType.SEQUENCE)))
    _raise_on_locator_user_type(locator)  # must not raise


# ---------------------------------------------------------------------------
# create_inputs_mapping
# ---------------------------------------------------------------------------


def test_create_inputs_mapping_args_go_to_input_args():
    loc = _locator("args_0")
    ts = Mock()
    ts.name = "args_0"
    input_args, input_kwargs = create_inputs_mapping(_mock_input_spec([(loc, ts)]))
    assert input_args == {"args_0": "args_0"}
    assert input_kwargs == {}


def test_create_inputs_mapping_kwargs_go_to_input_kwargs():
    loc = _locator("x", depth=1)
    ts = Mock()
    ts.name = "x"
    input_args, input_kwargs = create_inputs_mapping(_mock_input_spec([(loc, ts)]))
    assert input_args == {}
    assert ("x", 1) in input_kwargs
    assert input_kwargs[("x", 1)] == (loc, "x")


def test_create_inputs_mapping_mixed():
    loc_a = _locator("args_0")
    ts_a = Mock()
    ts_a.name = "args_0"
    loc_k = _locator("hidden_states", depth=1)
    ts_k = Mock()
    ts_k.name = "hidden_states"

    input_args, input_kwargs = create_inputs_mapping(_mock_input_spec([(loc_a, ts_a), (loc_k, ts_k)]))

    assert "args_0" in input_args
    assert ("hidden_states", 1) in input_kwargs


def test_create_inputs_mapping_nested_kwarg_stores_depth():
    """Nested kwargs (depth > 1) are stored with their actual depth as key."""
    loc = _locator("weight", depth=2)
    ts = Mock()
    ts.name = "weight"
    _, input_kwargs = create_inputs_mapping(_mock_input_spec([(loc, ts)]))
    assert ("weight", 2) in input_kwargs
    assert ("weight", 1) not in input_kwargs


# ---------------------------------------------------------------------------
# war_for_positional_arguments
# ---------------------------------------------------------------------------


def test_war_maps_positional_arg_to_kwarg_name():
    """Single positional arg gets mapped to the first forward_kwargs name."""
    input_args = {"args_0": "args_0"}
    forward_args = []
    forward_kwargs = ["x", "y"]

    war_for_positional_arguments(input_args, forward_args, forward_kwargs)

    assert input_args["x"] == "args_0"
    assert forward_args == ["x"]


def test_war_maps_multiple_positional_args():
    """Multiple positional args are all mapped in order."""
    input_args = {"args_0": "args_0", "args_1": "args_1"}
    forward_args = []
    forward_kwargs = ["x", "y", "z"]

    war_for_positional_arguments(input_args, forward_args, forward_kwargs)

    assert input_args["x"] == "args_0"
    assert input_args["y"] == "args_1"
    assert forward_args == ["x", "y"]


def test_war_stops_immediately_when_forward_args_present():
    """If true positional-only args exist in forward_args, nothing is remapped."""
    input_args = {"args_0": "args_0"}
    forward_args = ["x"]
    forward_kwargs = ["y"]

    war_for_positional_arguments(input_args, forward_args, forward_kwargs)

    assert "y" not in input_args
    assert forward_args == ["x"]


def test_war_no_op_when_input_args_empty():
    """Empty input_args → nothing to map, no mutation."""
    forward_args = []
    forward_kwargs = ["x"]

    war_for_positional_arguments({}, forward_args, forward_kwargs)

    assert forward_args == []


# ---------------------------------------------------------------------------
# create_ordered_dynamic_shapes
# ---------------------------------------------------------------------------


def test_create_ordered_dynamic_shapes_kwargs_only():
    """Pure-kwargs model → result dict in forward signature order."""
    dynamic_shapes = {"hidden_states": {0: "batch"}, "mask": {}}
    input_kwargs = {
        ("hidden_states", 1): (_locator("hidden_states", depth=1), "hidden_states"),
        ("mask", 1): (_locator("mask", depth=1), "mask"),
    }
    result = create_ordered_dynamic_shapes([], ["hidden_states", "mask"], {}, input_kwargs, dynamic_shapes)

    assert list(result.keys()) == ["hidden_states", "mask"]
    assert result["hidden_states"] == {0: "batch"}
    assert result["mask"] == {}


def test_create_ordered_dynamic_shapes_args_only():
    """Pure-args model → result dict keyed by forward_args names."""
    dynamic_shapes = {"x": {0: "batch"}}
    result = create_ordered_dynamic_shapes(["x"], [], {"x": "x"}, {}, dynamic_shapes)

    assert "x" in result
    assert result["x"] == {0: "batch"}


def test_create_ordered_dynamic_shapes_mixed_args_and_kwargs():
    """Mix of positional and keyword args → all included."""
    dynamic_shapes = {"x": {0: "batch"}, "mask": {}}
    input_kwargs = {("mask", 1): (_locator("mask", depth=1), "mask")}
    result = create_ordered_dynamic_shapes(["x"], ["mask"], {"x": "x"}, input_kwargs, dynamic_shapes)

    assert "x" in result
    assert "mask" in result


def test_create_ordered_dynamic_shapes_kwarg_not_in_spec_excluded():
    """Kwargs with no recorded input spec are omitted from the result."""
    dynamic_shapes = {"x": {0: "batch"}}
    input_kwargs = {("x", 1): (_locator("x", depth=1), "x")}
    result = create_ordered_dynamic_shapes([], ["x", "unused"], {}, input_kwargs, dynamic_shapes)

    assert "x" in result
    assert "unused" not in result
