# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Locator for an object in a nested structure."""

import dataclasses
from collections import deque
from collections.abc import Generator, Sequence
from enum import IntEnum
from typing import ClassVar


class ObjectType(IntEnum):
    """Object type."""

    SEQUENCE = 0
    DICT = 1
    DATACLASS = 2
    USER_TYPE = 3


class Locator:
    """A path-based locator for navigating and manipulating nested data structures.

    The Locator class provides mechanisms to identify and access elements (such as Python primitives,
    torch tensors, etc.) within nested structures—including sequences, dictionaries, dataclasses,
    and custom user types. It maintains a path that traverses the structure, allowing retrieval
    and mutation of values at those paths, as well as traversing nested contents and supporting
    custom types.

    The path is stored as a sequence of (accessor, ObjectType) tuples. The accessor is either
    an integer index (for sequences) or a string key/attribute name (for dicts, dataclasses,
    or user types).

    The primary entry point for this class is the `find_leaves` method, which yields all leaves
    found within a nested structure.

    Attributes:
        _user_types: Class-level set of registered user-defined types that should
            be traversed like dataclasses.

    """

    _user_types: ClassVar[set[type]] = set()

    def __init__(self, paths: Sequence[tuple[int | str, ObjectType]]):
        """Initialize the locator.

        Args:
            paths: Sequence of (accessor, ObjectType) tuples defining the path
                through a nested structure. The accessor is an int for sequences
                or a str for dicts/dataclasses/user types.
        """
        self._path = tuple(paths)  # make it immutable

    def __eq__(self, value: object, /) -> bool:
        """Equality operator."""
        if not isinstance(value, Locator):
            return False
        return self._path == value._path

    def __hash__(self) -> int:
        """Hash of the locator."""
        return hash(self._path)

    def __repr__(self):
        """Representation of the locator."""
        return self.__str__()

    def __str__(self):
        """String representation of the locator."""
        result = []
        for accessor, obj_type in self._path:
            if obj_type == ObjectType.DATACLASS or obj_type == ObjectType.USER_TYPE:
                result.append(f".{accessor}")
            elif obj_type == ObjectType.SEQUENCE:
                result.append(f"[{accessor}]")
            elif obj_type == ObjectType.DICT:
                result.append(f"['{accessor}']")
        return "".join(result)

    def get_value(self, obj: object) -> object:
        """Get the value of the locator."""
        for accessor, obj_type in self._path:
            if obj_type == ObjectType.DATACLASS or obj_type == ObjectType.USER_TYPE:
                obj = getattr(obj, accessor)
            else:
                obj = obj[accessor]

        return obj

    def set_value(self, obj: object, value: object) -> object:
        """Set the value of the locator."""
        num_paths = len(self._path)
        root = obj
        if num_paths == 0:
            return value
        for i, (accessor, obj_type) in enumerate(self._path):
            if i == num_paths - 1:
                break
            if obj_type == ObjectType.DATACLASS or obj_type == ObjectType.USER_TYPE:
                obj = getattr(obj, accessor)
            else:
                obj = obj[accessor]  # type: ignore
        if obj_type == ObjectType.DATACLASS or obj_type == ObjectType.USER_TYPE:
            setattr(obj, accessor, value)
        elif isinstance(obj, tuple):
            obj = list(obj)
            obj[accessor] = value  # type: ignore
            obj = tuple(obj)
            if i > 0:  # we have to modify parent with changed child reference
                parent_locator = Locator(self._path[:i])
                return parent_locator.set_value(root, obj)
            else:
                return obj
        else:
            obj[accessor] = value  # type: ignore
        return root

    @staticmethod
    def find_leaves(obj: object) -> Generator[tuple["Locator", object], None, None]:
        """Walks the object and yields locators and their leaf values.

        Leaf value is anything that is stored in a sequence, dictionary, or dataclass.
        """
        todo = deque([(obj, [])])
        while todo:
            current, stack = todo.popleft()
            if isinstance(current, (list, tuple)):
                for idx, value in enumerate(current):
                    todo.append((value, stack + [(idx, ObjectType.SEQUENCE)]))
            elif isinstance(current, dict):
                for key, value in current.items():
                    todo.append((value, stack + [(key, ObjectType.DICT)]))
            elif dataclasses.is_dataclass(current):
                for field in dataclasses.fields(current):
                    todo.append((getattr(current, field.name), stack + [(field.name, ObjectType.DATACLASS)]))
            elif type(current) in Locator._user_types:
                for field in get_object_fields(current):
                    todo.append((getattr(current, field), stack + [(field, ObjectType.USER_TYPE)]))
            else:
                yield Locator(stack), current

    @staticmethod
    def register_user_type(cls: type):
        """Register a user type."""
        Locator._user_types.add(cls)

    @staticmethod
    def unregister_user_type(cls: type):
        """Unregister a user type."""
        Locator._user_types.remove(cls)

    @staticmethod
    def is_user_type(cls: type) -> bool:
        """Check if a class is a user type."""
        return cls in Locator._user_types


def get_object_fields(obj):
    """Get only public data fields (excluding methods) including inherited ones."""
    cls = type(obj)

    # 2. Get all accessible attributes
    all_attributes = dir(obj)

    public_fields = []

    for attr in all_attributes:
        # Exclude names starting with '_'
        if attr.startswith("_"):
            continue

        # Get the actual attribute object from the class (not the instance)
        # This is safe because 'dir(obj)' only returns valid attributes.
        attr_value = getattr(cls, attr, None)

        # Check if the attribute is a property at the class level
        if isinstance(attr_value, property):
            continue

        # Check if the attribute is a regular method or other callable
        # We check the attribute from the instance here to catch methods
        # like unbound methods that return a bound method for an instance.
        if callable(getattr(obj, attr)):
            continue

        # If it passes all checks, it's considered a public data field
        public_fields.append(attr)

    return sorted(public_fields)
