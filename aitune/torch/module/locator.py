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

import torch


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
        _user_types: Class-level dict of registered user-defined types that should be traversed like dataclasses.
        If the value is True, only tensors are returned, otherwise all objects are returned.
        _ignored_types: Class-level set of types that should not be traversed nor returned by find_leaves.

    """

    _user_types: ClassVar[dict[type, bool]] = dict[type, bool]()
    _ignored_types: ClassVar[set[type]] = set()

    def __init__(self, paths: Sequence[tuple[int | str, ObjectType]]):
        """Initialize the locator.

        Args:
            paths: Sequence of (accessor, ObjectType) tuples defining the path
                through a nested structure. The accessor is an int for sequences
                or a str for dicts/dataclasses/user types.
        """
        self._path = tuple(paths)  # make it immutable

    @property
    def depth(self) -> int:
        """Get the depth of the locator."""
        return len(self._path)

    @property
    def root_name(self) -> int | str:
        """Get the root of the locator."""
        return self._path[0][0]

    @property
    def leaf_name(self) -> int | str:
        """Get the leaf of the locator."""
        return self._path[-1][0]

    def accessor_at(self, index: int) -> int | str:
        """Get the accessor at the given index."""
        return self._path[index][0]

    def path_iter(self):
        """Yield (accessor, ObjectType) tuples for each step in the path."""
        return iter(self._path)

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
    def _process_sequence(current: list | tuple, stack: list, only_tensors: bool, todo: deque):
        """Process a sequence and add its elements to the todo queue.

        Args:
            current: The sequence to process.
            stack: The current path stack.
            only_tensors: Whether to only process tensors.
            todo: The todo queue to append items to.
        """
        for idx, value in enumerate(current):
            todo.append((value, [*stack, (idx, ObjectType.SEQUENCE)], only_tensors))

    @staticmethod
    def _process_dict(current: dict, stack: list, only_tensors: bool, todo: deque):
        """Process a dictionary and add its items to the todo queue.

        Args:
            current: The dictionary to process.
            stack: The current path stack.
            only_tensors: Whether to only process tensors.
            todo: The todo queue to append items to.
        """
        for key, value in current.items():
            todo.append((value, [*stack, (key, ObjectType.DICT)], only_tensors))

    @staticmethod
    def _process_dataclass(current: object, stack: list, only_tensors: bool, todo: deque):
        """Process a dataclass and add its fields to the todo queue.

        Args:
            current: The dataclass to process.
            stack: The current path stack.
            only_tensors: Whether to only process tensors.
            todo: The todo queue to append items to.
        """
        for field in dataclasses.fields(current):  # type: ignore
            value = getattr(current, field.name)
            todo.append((value, [*stack, (field.name, ObjectType.DATACLASS)], only_tensors))

    @staticmethod
    def _process_user_type(current: object, stack: list, only_tensors: bool, todo: deque):
        """Process a user type and add its fields to the todo queue.

        Args:
            current: The user type object to process.
            stack: The current path stack.
            only_tensors: Whether to only process tensors.
            todo: The todo queue to append items to.
        """
        only_tensors = Locator._user_types[type(current)]
        for field in get_object_fields(current):
            value = getattr(current, field)
            todo.append((value, [*stack, (field, ObjectType.USER_TYPE)], only_tensors))

    @staticmethod
    def _should_ignore_object(current: object, only_tensors: bool) -> bool:
        """Check if the object should be ignored."""
        if type(current) in Locator._ignored_types:
            return True
        if only_tensors:
            return not torch.is_tensor(current)
        return False

    @staticmethod
    def find_leaves(obj: object, only_tensors: bool = False) -> Generator[tuple["Locator", object], None, None]:
        """Walks the object and yields locators and their leaf values.

        Leaf value is anything that is stored in a sequence, dictionary, or dataclass. If `only_tensors` is True,
        only tensors are returned, otherwise all objects are returned.

        Args:
            obj: The object to walk.
            only_tensors: Whether to only return tensors.

        Yields:
            A tuple of (locator, value) where locator is the path to the leaf and value is the leaf value.

        Note:
        - if a user type is marked as ignore, it will not be traversed nor returned.
        - if a user type is registered, it will be traversed like a dataclass. If it was registered with
              `only_tensors=True`, only tensors are returned irrespective of method argument `only_tensors`.
              Such a case may happen when we want to handle LLM cache and look only for tensors inside cache object.
        """
        todo = deque([(obj, [], only_tensors)])
        while todo:
            current, stack, only_tensors = todo.popleft()
            if isinstance(current, list | tuple):
                Locator._process_sequence(current, stack, only_tensors, todo)
            elif isinstance(current, dict):
                Locator._process_dict(current, stack, only_tensors, todo)
            elif dataclasses.is_dataclass(current):
                Locator._process_dataclass(current, stack, only_tensors, todo)
            elif type(current) in Locator._user_types:
                Locator._process_user_type(current, stack, only_tensors, todo)
            elif Locator._should_ignore_object(current, only_tensors):
                continue
            else:
                yield Locator(stack), current

    @staticmethod
    def register_user_type(cls: type, only_tensors: bool = False):
        """Register a user type."""
        Locator._user_types[cls] = only_tensors

    @staticmethod
    def unregister_user_type(cls: type):
        """Unregister a user type."""
        Locator._user_types.pop(cls, None)

    @staticmethod
    def ignore_type(cls: type):
        """Ignore a type."""
        Locator._ignored_types.add(cls)

    @staticmethod
    def unignore_type(cls: type):
        """Unignore a type."""
        Locator._ignored_types.discard(cls)


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
