# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rebuild structured backend outputs from a recipe prepared for a fixed graph.

Exported backends return tensors in recorded order, without the Python containers from the original
module output. GraphSpec selects this recipe when the output structure can be rebuilt without the
general SampleMetadata deepcopy and locator replacement steps. Preparation runs once; inference
fills prepared leaf positions and asks PyTree to rebuild the containers around the backend tensors.
"""

from typing import Any, cast

import torch
from torch.utils import _pytree

from aitune.torch.module.locator import ObjectType
from aitune.torch.module.sample_metadata import SampleMetadata

# Exact types exclude subclasses with mutable instance state or custom copy behavior.
_IMMUTABLE_LEAF_TYPES = (type(None), bool, int, float, complex, str, bytes, torch.dtype, torch.device)


class OutputReconstruction:
    """A reusable mapping from flat backend tensor results to one graph's Python output structure.

    Use prepare() to check compatibility and construct the recipe. GraphSpec caches the result for
    its fixed output template. On inference, exported backends call GraphSpec.restore_outputs();
    ForwardEffectsSpec applies input mutations, removes internal mutation results from the user
    output sequence, and delegates container reconstruction to unflatten_tensors(). Backends do
    not need to select or call this recipe directly.

    Preparation accepts exact built-in dict, list, and tuple containers with immutable non-tensor
    leaves. Each call creates fresh mutable containers and installs the current tensor references
    without copying tensor data. The recipe retains neither sample tensors nor previous results.
    Shared containers and custom objects keep the general SampleMetadata reconstruction path so
    their identity relationships and copy behavior remain intact. GraphSpec also excludes outputs
    with preserved input aliases, which must resolve caller-owned objects on each inference call.
    """

    # Previously pickled recipes have no collapsed scalar containers. An immutable class default
    # lets those recipes retain their original reconstruction without a per-call compatibility check.
    _scalar_container_templates: tuple[tuple[int, dict | list], ...] = ()

    def __init__(
        self,
        output_spec: SampleMetadata,
        tree_spec: _pytree.TreeSpec,
        leaf_template: tuple[Any, ...],
        tensor_leaf_indices: tuple[int, ...],
        scalar_container_templates: tuple[tuple[int, dict | list], ...] = (),
    ) -> None:
        """Store a recipe already checked by prepare().

        Args:
            output_spec: Metadata supplying tensor order and locators, without sample tensor values.
            tree_spec: Container structure produced by flattening the compatible template.
            leaf_template: Constants and placeholders to copy into a new leaf list for each call.
            tensor_leaf_indices: Leaf position for each recorded tensor; all positions must be resolved.
            scalar_container_templates: Leaf positions and independent scalar-container snapshots
                that require a fresh shallow copy on each call.
        """
        self._tensor_data = output_spec.tensor_data
        self._tree_spec = tree_spec
        self._leaf_template = leaf_template
        self._tensor_leaf_indices = tensor_leaf_indices
        self._scalar_container_templates = scalar_container_templates

    @classmethod
    def prepare(cls, output_spec: SampleMetadata, template: Any) -> "OutputReconstruction | None":
        """Check a fixed output template and prepare its tensor-to-leaf mapping once per graph.

        PyTree traversal order is derived from the template, whereas backends return tensors in
        output_spec order. Match full locator paths during preparation so inference can insert each
        tensor at its known leaf position without repeating path traversal or replacement grouping.

        Args:
            output_spec: Recorded user tensor paths and order, excluding internal mutation outputs.
            template: Fixed output structure, usually with tensor positions replaced by None.
                A template containing the original tensor references is also accepted, but those
                references are replaced by placeholders and are not retained in the recipe.

        Returns:
            A recipe if the structure and every recorded tensor path are supported. None selects
            the general SampleMetadata reconstruction path. This includes unsupported containers,
            incompatible leaves, and missing tensor positions. Complete flat tensor tuples also
            return None because SampleMetadata already returns their replacement tuple directly.
        """
        # A root tensor has no container tree to prepare. Custom root types must retain their
        # existing reconstruction and copy behavior, including named tuples and dataclasses.
        if type(template) not in (dict, list, tuple):
            return None
        tensor_paths = {tuple(locator.path_iter()): index for index, (locator, _) in enumerate(output_spec.tensor_data)}
        # Skip PyTree for tuples whose every slot is a tensor in recorded order: the existing
        # SampleMetadata fast path returns the validated tensor tuple without rebuilding it.
        if (
            type(template) is tuple
            and len(template) == len(tensor_paths)
            and all(tensor_paths.get(((index, ObjectType.SEQUENCE),)) == index for index in range(len(template)))
        ):
            return None
        # Check the whole tree before taking snapshots. A shared container or custom object
        # anywhere in the output requires the fallback to preserve identity and copy semantics.
        if not _supports_container_tree(template, set()):
            return None

        # Protect every tensor path and all its ancestors from scalar-container collapsing.
        # For {"tensors": [None, None], "metadata": [1, 2]}, recorded tensor paths protect
        # the root, "tensors", and its indexed slots. Otherwise the None placeholders would
        # make that list look scalar-only, hiding its tensor positions from PyTree traversal.
        # The unrelated "metadata" list can still be collapsed and copied as a single leaf.
        protected_paths = {path[:depth] for path in tensor_paths for depth in range(len(path) + 1)}
        scalar_templates = _prepare_scalar_containers(template, protected_paths)
        # Tensor-free scalar containers become single PyTree leaves. Copy them with the built-in
        # container operation per call instead of reconstructing every immutable scalar in Python.
        path_leaves, tree_spec = _pytree.tree_flatten_with_path(
            template, is_leaf=lambda value: id(value) in scalar_templates
        )
        leaves = []
        # -1 means this recorded tensor has not yet been matched to a template leaf. Each
        # successful path match replaces the sentinel with a position in the PyTree leaf list.
        tensor_leaf_indices = [-1] * len(output_spec.tensor_data)
        scalar_container_templates = []
        for key_path, value in path_leaves:
            path = tuple(
                (key.key, ObjectType.DICT)
                if isinstance(key, _pytree.MappingKey)
                else (cast(_pytree.SequenceKey, key).idx, ObjectType.SEQUENCE)
                for key in key_path
            )
            tensor_index = tensor_paths.get(path)
            if tensor_index is not None:
                # Use the recorded index rather than encounter order, and discard any sample
                # tensor reference so the reusable recipe cannot keep its storage alive.
                tensor_leaf_indices[tensor_index] = len(leaves)
                leaves.append(None)
            elif id(value) in scalar_templates:
                scalar_container_templates.append((len(leaves), scalar_templates[id(value)]))
                leaves.append(None)
            elif type(value) in _IMMUTABLE_LEAF_TYPES:
                leaves.append(value)
            else:
                # Includes tensors at unrecorded paths. The recipe must not capture them as
                # constants or silently drop values that the recorded contract cannot replace.
                return None
        # A remaining sentinel means a recorded tensor path did not resolve to a PyTree leaf.
        # Reject the recipe instead of using -1 as an index and overwriting the last leaf.
        if -1 in tensor_leaf_indices:
            return None
        return cls(output_spec, tree_spec, tuple(leaves), tuple(tensor_leaf_indices), tuple(scalar_container_templates))

    def unflatten_tensors(self, tensors: Any) -> Any:
        """Rebuild the user output from the current backend tensor results.

        This is the prepared alternative to SampleMetadata.unflatten_tensors(), selected inside
        ForwardEffectsSpec.restore_outputs() through GraphSpec.restore_outputs(). Exported
        TorchInductor AOT, Torch-TensorRT AOT, TensorRT, and ONNX Runtime backends use that shared
        restoration entry point. Input mutations have already been applied and extra mutation
        results excluded before this method runs. It handles container reconstruction only;
        preserved input aliases continue through SampleMetadata's general reconstruction path.

        Args:
            tensors: Iterable of user tensor results in recorded output_spec order. A bare tensor
                is also accepted when the recipe has exactly one recorded tensor output.

        Returns:
            The recorded Python output structure containing the supplied tensors by identity,
            with fresh mutable containers and isolated copies of non-tensor metadata.

        Raises:
            ValueError: The result count differs from the recorded user tensor count.
            TypeError: A recorded tensor position receives a non-tensor result.
        """
        if len(self._tensor_data) == 1 and isinstance(tensors, torch.Tensor):
            tensors = (tensors,)
        else:
            tensors = tuple(tensors)
        if len(tensors) != len(self._tensor_data):
            raise ValueError(f"Expected {len(self._tensor_data)} tensors, got {len(tensors)}.")

        # Work only on per-call state. Neither installing tensors nor copying metadata changes
        # the cached recipe, so later calls cannot observe this call's outputs or mutations.
        leaves = list(self._leaf_template)
        for (locator, _), index, tensor in zip(self._tensor_data, self._tensor_leaf_indices, tensors, strict=True):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Recorded tensor path {locator} received {type(tensor).__name__}, not a tensor.")
            leaves[index] = tensor
        for index, container in self._scalar_container_templates:
            leaves[index] = container.copy()
        return _pytree.tree_unflatten(leaves, self._tree_spec)


def _prepare_scalar_containers(
    template: Any, protected_paths: set[tuple[tuple[int | str, ObjectType], ...]]
) -> dict[int, dict | list]:
    """Find scalar-only lists and dictionaries that can be represented as single PyTree leaves.

    Called after _supports_container_tree() has excluded shared containers, cycles, and custom
    objects. A nonempty exact list or dict qualifies only when its immediate values have supported
    immutable types and its path is not protected. Nested containers are visited individually;
    a parent containing lists or dictionaries cannot be copied as a scalar-only container.

    Preparation snapshots each qualifying container instead of retaining the source object.
    Inference shallow-copies the snapshot, which isolates mutable container state while safely
    sharing immutable values. This avoids Python leaf-by-leaf reconstruction of large scalar lists
    and dictionaries. Empty containers remain in the ordinary PyTree structure.

    Args:
        template: Compatible output tree to inspect without modifying it.
        protected_paths: Recorded tensor paths and all their prefixes, using Locator path entries.
            Containers at these paths must stay visible to PyTree traversal even if their tensor
            placeholders are None and would otherwise satisfy the scalar-only check.

    Returns:
        Mapping from each qualifying source container's identity to its independent shallow-copy
        snapshot. Source identities are used only during preparation to recognize PyTree leaves;
        snapshots are later stored with their leaf positions for per-call copying.
    """
    templates = {}
    stack: list[tuple[tuple[tuple[int | str, ObjectType], ...], Any]] = [((), template)]
    while stack:
        path, value = stack.pop()
        if type(value) not in (dict, list, tuple):
            continue
        values = value.values() if type(value) is dict else value
        if (
            type(value) in (dict, list)
            and value
            and path not in protected_paths
            and all(type(child) in _IMMUTABLE_LEAF_TYPES for child in values)
        ):
            templates[id(value)] = value.copy()
            continue
        children = value.items() if type(value) is dict else enumerate(value)
        kind = ObjectType.DICT if type(value) is dict else ObjectType.SEQUENCE
        stack.extend(((*path, (key, kind)), child) for key, child in children)
    return templates


def _supports_container_tree(value: Any, seen: set[int]) -> bool:
    """Check whether ordinary PyTree rebuilding can preserve the template's container contract.

    Accept exact dict, list, and tuple containers, with exact int or str dictionary keys.
    Leaves may be tensors or the exact immutable types in _IMMUTABLE_LEAF_TYPES. Container and
    scalar subclasses are excluded because their instance state or copy hooks require the general
    reconstruction path. Tensor leaves are checked against recorded paths later in prepare().

    Track container identities across the entire traversal, not just the current recursion path.
    Seeing a container twice indicates sharing or a cycle. PyTree rebuilding would duplicate shared
    mutable containers and cannot represent cycles, so either case selects the general fallback.
    Repeated tensor references are allowed; recorded tensor positions determine their replacements.

    Args:
        value: Current subtree or leaf being checked.
        seen: Container identities encountered anywhere in this preparation traversal. The caller
            supplies an empty set for the root; recursive calls update and share that same set.

    Returns:
        True if every visited value uses supported types and every container occurs only once.
        False if custom types, unsupported keys, shared containers, or cycles require fallback.
    """
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
