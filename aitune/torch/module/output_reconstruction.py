# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare output containers once instead of copying and patching a template on every call."""

from typing import Any, cast

import torch
from torch.utils import _pytree

from aitune.torch.module.locator import ObjectType
from aitune.torch.module.sample_metadata import SampleMetadata

# Exact types exclude subclasses with mutable instance state or custom copy behavior.
_IMMUTABLE_LEAF_TYPES = (type(None), bool, int, float, complex, str, bytes, torch.dtype, torch.device)


class OutputReconstruction:
    """A prepared recipe for one fixed output template and its recorded tensor order.

    Preparation accepts built-in container trees with immutable non-tensor leaves. PyTree rebuilds
    fresh containers per call, without keeping sample tensors or traversing recorded locator paths.
    Shared containers and custom objects require the existing deepcopy/locator reconstruction so
    their identity and copy behavior remain intact. Preserved input aliases are handled by that
    existing path as well; this recipe accepts only ordinary tensor results.
    """

    def __init__(
        self,
        output_spec: SampleMetadata,
        tree_spec: _pytree.TreeSpec,
        leaf_template: tuple[Any, ...],
        tensor_leaf_indices: tuple[int, ...],
    ) -> None:
        """Store a checked structure with None at every tensor slot."""
        self._tensor_data = output_spec.tensor_data
        self._tree_spec = tree_spec
        self._leaf_template = leaf_template
        self._tensor_leaf_indices = tensor_leaf_indices

    @classmethod
    def prepare(cls, output_spec: SampleMetadata, template: Any) -> "OutputReconstruction | None":
        """Prepare compatible nested outputs; return None to retain the existing reconstruction.

        Root tensors and complete flat tensor tuples already have cheaper direct paths. Check
        eligibility once per graph, including shared-container detection and exact locator matching.
        """
        if type(template) not in (dict, list, tuple):
            return None
        tensor_paths = {tuple(locator.path_iter()): index for index, (locator, _) in enumerate(output_spec.tensor_data)}
        if (
            type(template) is tuple
            and len(template) == len(tensor_paths)
            and all(tensor_paths.get(((index, ObjectType.SEQUENCE),)) == index for index in range(len(template)))
        ):
            return None
        if not _supports_container_tree(template, set()):
            return None

        path_leaves, tree_spec = _pytree.tree_flatten_with_path(template)
        leaves = []
        tensor_leaf_indices = [-1] * len(output_spec.tensor_data)
        for key_path, value in path_leaves:
            path = tuple(
                (key.key, ObjectType.DICT)
                if isinstance(key, _pytree.MappingKey)
                else (cast(_pytree.SequenceKey, key).idx, ObjectType.SEQUENCE)
                for key in key_path
            )
            tensor_index = tensor_paths.get(path)
            if tensor_index is not None:
                tensor_leaf_indices[tensor_index] = len(leaves)
                leaves.append(None)
            elif type(value) in _IMMUTABLE_LEAF_TYPES:
                leaves.append(value)
            else:
                return None
        if -1 in tensor_leaf_indices:
            return None
        return cls(output_spec, tree_spec, tuple(leaves), tuple(tensor_leaf_indices))

    def unflatten_tensors(self, tensors: Any) -> Any:
        """Validate flat results and rebuild fresh containers around their tensor references."""
        if len(self._tensor_data) == 1 and isinstance(tensors, torch.Tensor):
            tensors = (tensors,)
        else:
            tensors = tuple(tensors)
        if len(tensors) != len(self._tensor_data):
            raise ValueError(f"Expected {len(self._tensor_data)} tensors, got {len(tensors)}.")

        leaves = list(self._leaf_template)
        for (locator, _), index, tensor in zip(self._tensor_data, self._tensor_leaf_indices, tensors, strict=True):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Recorded tensor path {locator} received {type(tensor).__name__}, not a tensor.")
            leaves[index] = tensor
        return _pytree.tree_unflatten(leaves, self._tree_spec)


def _supports_container_tree(value: Any, seen: set[int]) -> bool:
    """Reject custom copy behavior and shared containers that PyTree would duplicate."""
    if type(value) in (dict, list, tuple):
        if id(value) in seen:
            return False
        seen.add(id(value))
        if type(value) is dict:
            if any(type(key) not in (int, str) for key in value):
                return False
            children = value.values()
        else:
            children = value
        return all(_supports_container_tree(child, seen) for child in children)
    return type(value) in _IMMUTABLE_LEAF_TYPES or isinstance(value, torch.Tensor)
