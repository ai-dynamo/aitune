# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resize recorded samples to input shape boundaries."""

import copy
from typing import Any, cast

import torch

from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import Sample


def shape_boundary_samples(sample: Sample, graph_spec: GraphSpec, *, preserve_batch_axes: bool) -> tuple[Sample, ...]:
    """Prepare the strategy's minimum-first boundary group, or one unchanged static sample.

    The generic profiler does not infer shapes. Strategies opt into this preparation
    explicitly, keeping their minimum sample primary for reporting and saturation.
    """
    distinct_shapes = (
        any(spec.has_dynamic_axis() for spec in graph_spec.input_spec.tensor_specs)
        if preserve_batch_axes
        else graph_spec.input_spec.detected_dynamic_axis()
    )
    if not distinct_shapes:
        return (sample,)
    return (
        resize_sample_to_shape(sample, graph_spec, shape_attr="min_shape", preserve_batch_axes=preserve_batch_axes),
        resize_sample_to_shape(sample, graph_spec, shape_attr="max_shape", preserve_batch_axes=preserve_batch_axes),
    )


def resize_sample_to_shape(
    sample: tuple[tuple, dict],
    graph_spec: GraphSpec,
    *,
    shape_attr: str,
    preserve_batch_axes: bool = False,
) -> tuple[tuple, dict]:
    """Resize tensor inputs to recorded min/max shapes, preserving their call layout.

    preserve_batch_axes keeps the recorded batch dimensions unchanged when a
    later batch sweep will resize them. Without it, resizing to the maximum shape
    could expand batch axes before the requested batch is constructed.

    These synthetic values support shape validation and dense timing, not accuracy
    validation. Slicing and repeating can produce numerically invalid inputs and
    therefore NaN or Inf outputs.
    """
    args, kwargs = copy.deepcopy(sample)
    call_layout = graph_spec.forward_signature.capture_call_layout(args, kwargs)
    forward_inputs = graph_spec.forward_signature.normalize(args, kwargs)
    for locator, tensor_spec in graph_spec.input_spec.tensor_data:
        tensor = cast(torch.Tensor, locator.get_value(forward_inputs.arguments))
        target_shape = list(getattr(tensor_spec, shape_attr))
        if preserve_batch_axes:
            for axis in tensor_spec.get_batch_axis_multipliers():
                target_shape[axis] = tensor.shape[axis]
        resized_tensor = _resize_tensor(tensor, target_shape)
        updated_arguments = locator.set_value(forward_inputs.arguments, resized_tensor)
        forward_inputs.arguments = cast(dict[str, Any], updated_arguments)
    return call_layout.restore(forward_inputs.arguments)


def _resize_tensor(tensor: torch.Tensor, target_shape: list[int]) -> torch.Tensor:
    """Resize each axis and fill with zeros when expanding an empty tensor.

    Note: These samples are intended only for validating min and max shapes. Their tensor values may be
    numerically invalid and can produce NaN or Inf outputs.
    """
    for axis, target_size in enumerate(target_shape):
        current_size = tensor.shape[axis]
        if current_size == target_size:
            continue
        if current_size == 0:
            return tensor.new_zeros(target_shape)
        slice_indices = [slice(None)] * tensor.ndim
        if current_size > target_size:
            slice_indices[axis] = slice(target_size)
            tensor = tensor[tuple(slice_indices)]
            continue

        slice_indices[axis] = slice(1)
        tensor = tensor[tuple(slice_indices)]
        repeat_indices = [1] * tensor.ndim
        repeat_indices[axis] = target_size
        tensor = tensor.repeat(repeat_indices)
    return tensor
