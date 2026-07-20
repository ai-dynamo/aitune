# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aitune.torch.utils.shapes."""

import pytest

from aitune.torch.module.locator import Locator, ObjectType
from aitune.torch.utils.shapes import _create_nested_structure, _raise_on_locator_user_type

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
