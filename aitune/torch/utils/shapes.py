# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shapes utilities."""

import logging
from itertools import zip_longest
from typing import Any

import torch

from aitune.torch.module.locator import Locator, ObjectType

# Setup logger
logger = logging.getLogger(__name__)


def create_dynamic_shapes_from_tensor_spec(
    tensor_specs,
    _fill_missing_dims: bool = False,
    _fill_missing_dims_with: Any = None,
    _use_cached_dims: bool = False,
    use_auto: bool = False,
) -> dict[str, Any]:
    """Produces flat structure of dynamic shapes, tensor_spec name to dynamic shapes.

    Args:
        tensor_specs: List of tensor specs
        use_auto: When True, use ``torch.export.Dim.AUTO`` for all varying axes instead of
            explicit min/max Dims.  This lets torch.export infer divisibility constraints
            automatically, avoiding ``ConstraintViolationError`` when the model's internal
            guards (e.g. stride-2 conv divisibility) are not satisfied by the full [min, max]
            range.  Should be ``True`` for dynamo-based export paths.

        # Experimental arguments
        _fill_missing_dims: Whether to non dynamic dimensions
        _fill_missing_dims_with: what to fill missing dimensions with
        other options:
            torch.export.Dim.STATIC
            torch.export.Dim.AUTO
        _use_cached_dims: use cached dimensions across input tensors (ignored when use_auto=True)
    """
    dynamic_shapes = {}
    dim_cache = {}
    for tensor_spec in tensor_specs:
        dynamic_shapes[tensor_spec.name] = {}
        batch_axes = set(tensor_spec.get_batch_axis_multipliers().keys()) if use_auto else set()
        for idx, (d1, d2) in enumerate(zip(tensor_spec.min_shape, tensor_spec.max_shape, strict=True)):
            if d1 != d2:
                # Use Dim.AUTO only for non-batch varying axes: batch axes have no
                # divisibility constraints and explicit min/max is required for
                # ONNX shape inference (e.g. onnxruntime.quantization) to work correctly.
                if use_auto and idx not in batch_axes:
                    dim = torch.export.Dim.AUTO
                else:
                    dim = torch.export.Dim(f"{tensor_spec.name}_dim_{idx}", min=d1, max=d2)
                    if _use_cached_dims:
                        key = (d1, d2)
                        dim = dim_cache.setdefault(key, dim)

                dynamic_shapes[tensor_spec.name][idx] = dim
            elif _fill_missing_dims:
                dynamic_shapes[tensor_spec.name][idx] = _fill_missing_dims_with
    return dynamic_shapes


def create_inputs_mapping(input_spec) -> tuple[dict, dict]:
    """Creates yet another input names mapping but with depth of nested tensors.

    NOTE: NOT using graph_spec.input_spec.get_names_mapping() as we require depth of nested objects
    """
    input_args, input_kwargs = {}, {}
    for locator, tensor_spec in input_spec.tensor_data:
        if tensor_spec.name.startswith("args"):
            input_args[locator.leaf_name] = tensor_spec.name
            continue

        # saving key with depth to avoid ambiguity with nested objects
        input_kwargs[(locator.leaf_name, locator.depth)] = (locator, tensor_spec.name)
    return input_args, input_kwargs


def war_for_positional_arguments(input_args, forward_args, forward_kwargs):
    """Checks if we got positional argument that python marked as kwarg and adds it to input_args.

    It happens when argument is passed as positional and function signature does not have clear separation
    between positional and keyword arguments like with / separator: def forward(x, y, z, w, /): return x, y, z, w

    Name of the argument is stored in leaf of the locator.
    """
    for i, (arg_n, arg_name) in enumerate(zip_longest(list(input_args.items()), list(forward_args), fillvalue=None)):
        if arg_name is not None:
            # assuming it is ok
            break
        arg_name = forward_kwargs[i]
        forward_args.append(arg_name)
        input_args[arg_name] = arg_n[1]


def create_ordered_dynamic_shapes(forward_args, forward_kwargs, input_args, input_kwargs, dynamic_shapes) -> dict:
    """Creates ordered dynamic shapes with python function signature argument names."""
    ordered_dynamic_shapes = {}
    for fwd_arg in forward_args:
        if tensor_spec_name := input_args.get(fwd_arg):
            ordered_dynamic_shapes[fwd_arg] = dynamic_shapes[tensor_spec_name]

    for kwarg in forward_kwargs:
        if loc_and_name := input_kwargs.get((kwarg, 1)):
            _locator, tensor_spec_name = loc_and_name
            ordered_dynamic_shapes[kwarg] = dynamic_shapes[tensor_spec_name]

    # by utilizing locator we can create a structure of nested dynamic shapes
    for _, (locator, tensor_spec_name) in input_kwargs.items():
        if locator.depth > 1:
            # need to provide a parents as they are not created by set_value
            _raise_on_locator_user_type(locator)
            ordered_dynamic_shapes = _create_nested_structure(locator, root=ordered_dynamic_shapes)
            locator.set_value(ordered_dynamic_shapes, dynamic_shapes[tensor_spec_name])

    return ordered_dynamic_shapes


def print_dynamic_shapes(dynamic_shapes: dict) -> None:
    """Print the dynamic shapes."""
    logger.debug("Extracted dynamic shapes:")
    for key, value in dynamic_shapes.items():
        logger.debug("%s:", key)
        for idx, ds in (value or {}).items():
            if isinstance(ds, torch.export.dynamic_shapes._Dim):
                logger.debug("  %s: min:%d, max:%d, %s", idx, ds.min, ds.max, str(ds).split(".")[-1])
            else:
                logger.debug("  %s: %s", idx, ds)


def _create_nested_structure(locator: Locator, root: Any = None, last: Any = None) -> Any:
    """As locator does not create parent objects, we need to create them manually.

    Args:
        locator: Locator object
        root: Root object in the path
        last: Last object in the path
    Returns:
        Root object in the path
    """
    parents = [[] if T == ObjectType.SEQUENCE else {} for _, T in locator.path_iter()]
    accessors = [a for a, _ in locator.path_iter()]

    if root is not None:
        parents[0] = root

    parent = parents[0]
    for accessor, current in zip(accessors, parents[1:] + [last], strict=True):
        if isinstance(parent, list) and len(parent) <= accessor:
            parent.extend([None] * (accessor - len(parent) + 1))

        if accessor in parent:
            current = parent[accessor]

        parent[accessor] = current
        parent = current

    return parents[0]


def _raise_on_locator_user_type(locator: Locator) -> None:
    """Raise an exception if the locator is a user type."""
    if any(T in [ObjectType.USER_TYPE, ObjectType.DATACLASS] for _, T in locator.path_iter()):
        raise ValueError("Dynamic shapes does not support user types or dataclasses.")
