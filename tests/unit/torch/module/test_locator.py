# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from collections import UserDict

import torch

from aitune.torch.module.locator import Locator, ObjectType


class UserType:
    """Test user type for locator testing."""

    def __init__(self, value):
        self.value = value
        self._hidden = None

    @property
    def prop(self):
        """Should not be included in the locator."""
        return self._hidden

    @prop.setter
    def should_not_be_included(self, value):
        """Should not be included in the locator."""
        self._hidden = value

    def public_method(self):
        """Should not be included in the locator."""
        return self._hidden


@dataclasses.dataclass
class InnerDataClass:
    """Test dataclass for locator testing."""

    a: int = 1
    b: list = dataclasses.field(default_factory=lambda: [2, 3])
    c: dict = dataclasses.field(default_factory=lambda: {"d": 4})


@dataclasses.dataclass
class NestedDataClass:
    """Nested dataclass for complex testing."""

    inner: InnerDataClass = dataclasses.field(default_factory=InnerDataClass)
    value: int = 5


def test_locator_string_representation_sequence():
    # Test list access
    locator = Locator(((0, ObjectType.SEQUENCE),))
    assert str(locator) == "[0]"

    # Test nested list access
    locator = Locator(((0, ObjectType.SEQUENCE), (1, ObjectType.SEQUENCE)))
    assert str(locator) == "[0][1]"


def test_locator_string_representation_dict():
    # Test dict access
    locator = Locator((("key", ObjectType.DICT),))
    assert str(locator) == "['key']"

    # Test nested dict access
    locator = Locator((("a", ObjectType.DICT), ("b", ObjectType.DICT)))
    assert str(locator) == "['a']['b']"


def test_locator_string_representation_dataclass():
    # Test dataclass access
    locator = Locator((("field", ObjectType.DATACLASS),))
    assert str(locator) == ".field"

    # Test nested dataclass access
    locator = Locator((("outer", ObjectType.DATACLASS), ("inner", ObjectType.DATACLASS)))
    assert str(locator) == ".outer.inner"


def test_locator_string_representation_user_type():
    locator = Locator((("field", ObjectType.USER_TYPE),))
    assert str(locator) == ".field"


def test_locator_string_representation_mixed():
    locator = Locator((("field", ObjectType.DATACLASS), (0, ObjectType.SEQUENCE), ("key", ObjectType.DICT)))
    assert str(locator) == ".field[0]['key']"


def test_find_leaves_primitive_value():
    results = list(Locator.find_leaves("Abc"))
    assert len(results) == 1
    locator, value = results[0]
    assert value == "Abc"
    assert locator.get_value("Abc") == "Abc"
    assert str(locator) == ""


def test_find_leaves_nested_sequence():
    obj = [1, (2, 3)]
    results = list(Locator.find_leaves(obj))

    expected = [(1, "[0]"), (2, "[1][0]"), (3, "[1][1]")]

    assert len(results) == 3
    for (locator, value), (expected_value, expected_str) in zip(results, expected, strict=True):
        assert value == expected_value
        assert locator.get_value(obj) == expected_value
        assert str(locator) == expected_str


def test_find_leaves_nested_dict():
    obj = {"a": 1, "b": {"c": 2, "d": 3}}
    results = list(Locator.find_leaves(obj))

    expected = [(1, "['a']"), (2, "['b']['c']"), (3, "['b']['d']")]

    assert len(results) == 3
    for (locator, value), (expected_value, expected_str) in zip(results, expected, strict=True):
        assert value == expected_value
        assert locator.get_value(obj) == expected_value
        assert str(locator) == expected_str


def test_find_leaves_nested_dataclass():
    """Test traversal of nested dataclasses."""
    obj = NestedDataClass()
    results = sorted(Locator.find_leaves(obj), key=lambda x: x[1])

    expected = [(1, ".inner.a"), (2, ".inner.b[0]"), (3, ".inner.b[1]"), (4, ".inner.c['d']"), (5, ".value")]

    assert len(results) == 5
    for (locator, value), (expected_value, expected_str) in zip(results, expected, strict=True):
        assert value == expected_value
        assert locator.get_value(obj) == expected_value
        assert str(locator) == expected_str


def test_find_leaves_unregistered_user_type():
    # test unregistered user type - find leaves should return the object itself
    obj = UserType(1)
    results = list(Locator.find_leaves(obj))
    assert len(results) == 1
    locator, value = results[0]
    assert value == obj, "object should not be traversed"
    assert str(locator) == ""


def test_find_leaves_registered_user_type():
    # test registered user type - find leaves should traverse the object
    try:
        obj = UserType(1), "not_important"
        Locator.register_user_type(UserType, only_tensors=False)
        results = list(Locator.find_leaves(obj))
        assert len(results) == 2
        locator, value = results[0]
        assert value == 1, "object should be traversed"
        assert str(locator) == "[0].value"
        locator, value = results[1]
        assert value == "not_important", "object should be traversed"
        assert str(locator) == "[1]"

        Locator.register_user_type(UserType, only_tensors=True)
        results = list(Locator.find_leaves(obj))
        assert len(results) == 1, "object should not return any non tensor leaves"

        obj = UserType(torch.tensor(1))
        Locator.register_user_type(UserType, only_tensors=True)
        results = list(Locator.find_leaves(obj))
        assert len(results) == 1, "object should return a tensor leaf"
        locator, value = results[0]
        assert value == torch.tensor(1), "object should return a tensor leaf"
        assert str(locator) == ".value"

        # more complicated example to test scenario where we globally look for any objects but narrow search
        # for user type objects to only return tensor leaves e.g. for LLM cache we look only for tensors
        results = list(Locator.find_leaves(("abc", torch.tensor(2), obj), only_tensors=False))
        assert len(results) == 3
        locator, value = results[0]
        assert value == "abc", "non tensor leaf should be returned"
        assert str(locator) == "[0]"
        locator, value = results[1]
        assert value == torch.tensor(2), "tensor leaf should be returned"
        assert str(locator) == "[1]"
        locator, value = results[2]
        assert value == torch.tensor(1), "user type object should be traversed for tensor only"
        assert str(locator) == "[2].value"
    finally:
        Locator.unregister_user_type(UserType)


def test_find_leaves_ignored_type():
    obj = UserType(1)
    # check type is found if not ignored
    results = list(Locator.find_leaves(obj))
    assert len(results) == 1
    # now ignore the type
    Locator.ignore_type(UserType)
    results = list(Locator.find_leaves(obj))
    try:
        assert len(results) == 0
    finally:
        Locator.unignore_type(UserType)


def test_find_leaves_empty_containers():
    """Test traversal of empty containers."""
    results = list(Locator.find_leaves([]))
    assert len(results) == 0

    results = list(Locator.find_leaves({}))
    assert len(results) == 0


def test_set_value_list():
    obj = [1, 2, 3]
    results = list(Locator.find_leaves(obj))

    first_locator, _ = results[0]
    new_obj = first_locator.set_value(obj, -1)
    assert new_obj == [-1, 2, 3]

    second_locator, _ = results[1]
    new_obj = second_locator.set_value(new_obj, -2)
    assert new_obj == [-1, -2, 3]

    third_locator, _ = results[2]
    new_obj = third_locator.set_value(new_obj, -3)
    assert new_obj == [-1, -2, -3]


def test_set_value_nested_list():
    obj = [1, [2, 3]]
    results = list(Locator.find_leaves(obj))

    first_locator, _ = results[0]
    new_obj = first_locator.set_value(obj, -1)
    assert new_obj == [-1, [2, 3]]

    second_locator, _ = results[1]
    new_obj = second_locator.set_value(new_obj, -2)
    assert new_obj == [-1, [-2, 3]]

    third_locator, _ = results[2]
    new_obj = third_locator.set_value(new_obj, -3)
    assert new_obj == [-1, [-2, -3]]


def test_set_value_tuple():
    obj = (1, 2, 3)
    results = list(Locator.find_leaves(obj))

    first_locator, _ = results[0]
    new_obj = first_locator.set_value(obj, -1)
    assert new_obj == (-1, 2, 3)

    second_locator, _ = results[1]
    new_obj = second_locator.set_value(new_obj, -2)
    assert new_obj == (-1, -2, 3)

    third_locator, _ = results[2]
    new_obj = third_locator.set_value(new_obj, -3)
    assert new_obj == (-1, -2, -3)


def test_set_value_nested_tuple():
    obj = (1, (2, 3))
    results = list(Locator.find_leaves(obj))

    first_locator, _ = results[0]
    new_obj = first_locator.set_value(obj, -1)
    assert new_obj == (-1, (2, 3))

    second_locator, _ = results[1]
    new_obj = second_locator.set_value(new_obj, -2)
    assert new_obj == (-1, (-2, 3))

    third_locator, _ = results[2]
    new_obj = third_locator.set_value(new_obj, -3)
    assert new_obj == (-1, (-2, -3))


def test_set_value_dict():
    obj = {"a": 1, "b": {"c": 2, "d": 3}}
    results = list(Locator.find_leaves(obj))

    first_locator, _ = results[0]
    new_obj = first_locator.set_value(obj, -1)
    assert new_obj == {"a": -1, "b": {"c": 2, "d": 3}}

    second_locator, _ = results[1]
    new_obj = second_locator.set_value(new_obj, -2)
    assert new_obj == {"a": -1, "b": {"c": -2, "d": 3}}

    third_locator, _ = results[2]
    new_obj = third_locator.set_value(new_obj, -3)
    assert new_obj == {"a": -1, "b": {"c": -2, "d": -3}}


def test_locator_equality():
    locator1 = Locator(((0, ObjectType.SEQUENCE),))
    locator2 = Locator(((0, ObjectType.SEQUENCE),))
    assert locator1 == locator2

    locator1 = Locator(((0, ObjectType.SEQUENCE),))
    locator2 = Locator(((1, ObjectType.SEQUENCE),))
    assert locator1 != locator2

    locator1 = Locator(((0, ObjectType.SEQUENCE),))
    locator2 = Locator(((0, ObjectType.DICT),))
    assert locator1 != locator2


def test_hash():
    locator1 = Locator(((0, ObjectType.SEQUENCE),))
    locator2 = Locator(((0, ObjectType.SEQUENCE),))
    assert hash(locator1) == hash(locator2)

    locator1 = Locator(((0, ObjectType.SEQUENCE),))
    locator2 = Locator(((1, ObjectType.SEQUENCE),))
    assert hash(locator1) != hash(locator2)

    locator1 = Locator(((0, ObjectType.SEQUENCE),))
    locator2 = Locator((("key", ObjectType.DICT),))
    assert hash(locator1) != hash(locator2)


def test_locator_depth():
    locator = Locator(((0, ObjectType.SEQUENCE),))
    assert locator.depth == 1

    locator = Locator(((0, ObjectType.SEQUENCE), (0, ObjectType.SEQUENCE)))
    assert locator.depth == 2

    locator = Locator(((0, ObjectType.SEQUENCE), (0, ObjectType.SEQUENCE), (0, ObjectType.SEQUENCE)))
    assert locator.depth == 3


def test_locator_accessor_at():
    locator = Locator(((0, ObjectType.SEQUENCE),))
    assert locator.accessor_at(0) == 0

    locator = Locator((("key", ObjectType.DICT), (0, ObjectType.SEQUENCE)))
    assert locator.accessor_at(0) == "key"
    assert locator.accessor_at(1) == 0

    locator = Locator((("field", ObjectType.DATACLASS), (0, ObjectType.SEQUENCE), ("key", ObjectType.DICT)))
    assert locator.accessor_at(0) == "field"
    assert locator.accessor_at(1) == 0
    assert locator.accessor_at(2) == "key"


def test_locator_path_iter():
    locator = Locator(())
    assert list(locator.path_iter()) == []

    locator = Locator(((0, ObjectType.SEQUENCE),))
    assert list(locator.path_iter()) == [(0, ObjectType.SEQUENCE)]

    locator = Locator((("key", ObjectType.DICT), (0, ObjectType.SEQUENCE)))
    assert list(locator.path_iter()) == [("key", ObjectType.DICT), (0, ObjectType.SEQUENCE)]

    locator = Locator((("field", ObjectType.DATACLASS), (0, ObjectType.SEQUENCE), ("key", ObjectType.DICT)))
    assert list(locator.path_iter()) == [
        ("field", ObjectType.DATACLASS),
        (0, ObjectType.SEQUENCE),
        ("key", ObjectType.DICT),
    ]


def test_locator_root_name():
    locator = Locator(())
    assert locator.root_name == ""

    locator = Locator(((0, ObjectType.SEQUENCE),))
    assert locator.root_name == 0

    locator = Locator((("key", ObjectType.DICT), (0, ObjectType.SEQUENCE)))
    assert locator.root_name == "key"

    locator = Locator((("field", ObjectType.DATACLASS), (0, ObjectType.SEQUENCE), ("key", ObjectType.DICT)))
    assert locator.root_name == "field"


def test_locator_leaf_name():
    locator = Locator(((0, ObjectType.SEQUENCE),))
    assert locator.leaf_name == 0

    locator = Locator((("key", ObjectType.DICT), (0, ObjectType.SEQUENCE)))
    assert locator.leaf_name == 0

    locator = Locator((("field", ObjectType.DATACLASS), (0, ObjectType.SEQUENCE), ("key", ObjectType.DICT)))
    assert locator.leaf_name == "key"


class TensorUserDict(UserDict):
    """Minimal UserDict subclass mimicking BatchEncoding shape."""

    pass


def test_find_leaves_user_dict_non_strict():
    """UserDict subclass is traversed; tensors found, non-tensors skipped."""
    t1 = torch.tensor([1, 2, 3])
    t2 = torch.tensor([4, 5, 6])
    obj = TensorUserDict({"input_ids": t1, "attention_mask": t2, "meta": "some_string"})

    results = list(Locator.find_leaves(obj, only_tensors=True))

    assert len(results) == 2
    locators = {str(loc): val for loc, val in results}
    assert "['input_ids']" in locators
    assert "['attention_mask']" in locators
    assert torch.equal(locators["['input_ids']"], t1)
    assert torch.equal(locators["['attention_mask']"], t2)


def test_find_leaves_user_dict_all_leaves():
    """UserDict subclass is traversed; all leaves returned when only_tensors=False."""
    t = torch.tensor([1, 2, 3])
    obj = TensorUserDict({"input_ids": t, "meta": "some_string"})

    results = list(Locator.find_leaves(obj, only_tensors=False))

    assert len(results) == 2
    locators = {str(loc): val for loc, val in results}
    assert "['input_ids']" in locators
    assert "['meta']" in locators
    assert torch.equal(locators["['input_ids']"], t)
    assert locators["['meta']"] == "some_string"


def test_find_leaves_user_dict_nested_inside_dict():
    """UserDict nested inside a plain dict is fully traversed."""
    t = torch.tensor([7, 8])
    obj = {"outer": TensorUserDict({"input_ids": t, "flag": True})}

    results = list(Locator.find_leaves(obj, only_tensors=False))

    assert len(results) == 2
    locators = {str(loc): val for loc, val in results}
    assert "['outer']['input_ids']" in locators
    assert "['outer']['flag']" in locators
    assert torch.equal(locators["['outer']['input_ids']"], t)
    assert locators["['outer']['flag']"] is True


def test_find_leaves_user_dict_locator_get_value():
    """Locators returned for UserDict leaves correctly retrieve values via get_value."""
    t = torch.tensor([1, 2])
    obj = TensorUserDict({"input_ids": t})

    results = list(Locator.find_leaves(obj, only_tensors=False))
    assert len(results) == 1
    locator, _ = results[0]
    assert torch.equal(locator.get_value(obj), t)
    assert str(locator) == "['input_ids']"
