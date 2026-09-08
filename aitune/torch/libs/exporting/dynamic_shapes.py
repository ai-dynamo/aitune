# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build dynamic shape rules and safe example inputs for exporters."""

import logging
from copy import deepcopy
from typing import Any

import torch

from aitune.torch.dynamic_shapes import BatchDim, ShapeDefinition
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import Sample

logger = logging.getLogger(__name__)

TorchDynamicShapes = dict[str, tuple[dict[int, Any], ...]]


def log_dynamic_shapes(dynamic_shapes: dict[str, Any] | None) -> None:
    """Write each dynamic shape rule to the debug log."""
    if dynamic_shapes is None:
        return

    logger.debug("Extracted dynamic shapes:")

    def _print(value: Any, path: str) -> None:
        """Log each value in a nested shape rule with its full path."""
        if isinstance(value, dict):
            for key, child in value.items():
                _print(child, f"{path}[{key!r}]")
        elif isinstance(value, list | tuple):
            for index, child in enumerate(value):
                _print(child, f"{path}[{index}]")
        elif isinstance(value, torch.export.dynamic_shapes._Dim):
            logger.debug("  %s: min:%d, max:%d, %s", path, value.min, value.max, str(value).split(".")[-1])
        elif value is not None:
            logger.debug("  %s: %s", path, value)

    _print(dynamic_shapes, "dynamic_shapes")


def make_batch_dim(tensor_specs) -> Any:
    """Create one shared export dimension for all recorded batch axes."""
    if not any(ts.has_batch_axis() for ts in tensor_specs):
        return None
    min_val = float("inf")
    max_val = 0
    for tensor_spec in tensor_specs:
        for axis, multiplier in tensor_spec.get_batch_axis_multipliers().items():
            min_val = min(min_val, tensor_spec.min_shape[axis] // multiplier)
            max_val = max(max_val, tensor_spec.max_shape[axis] // multiplier)
    if min_val == max_val:
        return None
    return torch.export.Dim("batch", min=int(min_val), max=int(max_val))


def axis_dims_for_tensor(
    tensor_spec,
    batch_dim: Any,
    *,
    tensor_name: str,
    use_auto: bool = True,
    dim_cache: dict | None = None,
) -> dict[int, Any]:
    """Build export rules for the dynamic axes of one recorded tensor."""
    axis_dims: dict[int, Any] = {}
    if batch_dim is not None:
        for axis, multiplier in tensor_spec.get_batch_axis_multipliers().items():
            if use_auto:
                axis_dims[axis] = torch.export.Dim.AUTO
            else:
                axis_dims[axis] = batch_dim if multiplier == 1 else multiplier * batch_dim
    for index, entry in enumerate(tensor_spec.shape):
        if isinstance(entry, str) and entry.startswith("dim"):
            if tensor_spec.min_shape[index] != tensor_spec.max_shape[index]:
                if use_auto:
                    axis_dims[index] = torch.export.Dim.AUTO
                else:
                    key = (tensor_spec.min_shape[index], tensor_spec.max_shape[index])
                    if dim_cache is not None and key in dim_cache:
                        axis_dims[index] = dim_cache[key]
                    else:
                        dim = torch.export.Dim(f"{tensor_name}_dim_{index}", min=key[0], max=key[1])
                        if dim_cache is not None:
                            dim_cache[key] = dim
                        axis_dims[index] = dim
    return axis_dims


def axis_dims_for_shape_definition(
    definition: ShapeDefinition,
    dim_cache: dict[str, Any],
    *,
    use_auto: bool = True,
) -> dict[int, Any]:
    """Build export rules from a shape definition provided by the user."""
    axis_dims = {}
    for axis, dimension in enumerate(definition):
        if isinstance(dimension, int):
            continue
        if use_auto and not isinstance(dimension, BatchDim):
            axis_dims[axis] = torch.export.Dim.DYNAMIC(min=dimension.min, max=dimension.max)
            continue
        if dimension.name not in dim_cache:
            dim_cache[dimension.name] = torch.export.Dim(
                dimension.name,
                min=dimension.min,
                max=dimension.max,
            )
        axis_dims[axis] = dim_cache[dimension.name]
    return axis_dims


def build_dynamic_shapes(
    graph_spec: GraphSpec,
    *,
    use_auto: bool = True,
) -> TorchDynamicShapes | None:
    """Build constraints for the flat tensor boundary passed to ``torch.export``."""
    input_spec = graph_spec.input_spec
    inferred_tensor_specs = [
        tensor_spec
        for locator, tensor_spec in input_spec.tensor_data
        if graph_spec.get_shape_definition(locator) is None
    ]
    batch_dim = make_batch_dim(inferred_tensor_specs)
    dim_cache: dict | None = None if use_auto else {}
    explicit_dim_cache: dict[str, Any] = {}

    result = []
    has_dynamic_axes = False
    for locator, tensor_spec in input_spec.tensor_data:
        definition = graph_spec.get_shape_definition(locator)
        if definition is not None:
            dims = axis_dims_for_shape_definition(definition, explicit_dim_cache, use_auto=use_auto)
        elif tensor_spec.has_batch_axis() or tensor_spec.has_dynamic_axis():
            dims = axis_dims_for_tensor(
                tensor_spec,
                batch_dim,
                tensor_name=locator.display_path,
                use_auto=use_auto,
                dim_cache=dim_cache,
            )
        else:
            dims = {}
        result.append(dims)
        has_dynamic_axes |= bool(dims)

    return {"flat_inputs": tuple(result)} if has_dynamic_axes else None


def prepare_export_sample(sample: Sample, graph_spec: GraphSpec) -> Sample:
    """Prepare a sample that keeps dynamic axes visible to Torch Export."""
    input_spec = graph_spec.input_spec

    if input_spec.has_batch_axis():
        max_batch_size = graph_spec.get_max_batch_size()
        batch_size = min(max_batch_size, 2)
        args, kwargs = graph_spec.make_batch(*sample, batch_size=batch_size)
    else:
        args, kwargs = deepcopy(sample)

    call_layout = graph_spec.forward_signature.capture_call_layout(args, kwargs)
    forward_inputs = graph_spec.forward_signature.normalize(args, kwargs)
    for locator, tensor_spec in input_spec.tensor_data:
        try:
            tensor = locator.get_value(forward_inputs.arguments)
        except (IndexError, KeyError):
            continue
        if not isinstance(tensor, torch.Tensor):
            continue
        modified = False
        definition = graph_spec.get_shape_definition(locator)
        if definition is not None:
            dynamic_axes = [axis for axis, dimension in enumerate(definition) if not isinstance(dimension, int)]
        else:
            dynamic_axes = [
                axis
                for axis, entry in enumerate(tensor_spec.shape)
                if (
                    isinstance(entry, str)
                    and entry.startswith("dim")
                    and tensor_spec.min_shape[axis] < tensor_spec.max_shape[axis]
                    and tensor_spec.max_shape[axis] >= 2
                )
            ]
        for axis in dynamic_axes:
            if tensor.shape[axis] <= 1:
                tensor = torch.cat([tensor, tensor], dim=axis)
                modified = True
        if modified:
            forward_inputs.arguments = locator.set_value(forward_inputs.arguments, tensor)
    return call_layout.restore(forward_inputs.arguments)
