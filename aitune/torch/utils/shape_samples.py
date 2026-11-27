# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Construct synthetic inputs from recorded shape bounds."""

from copy import deepcopy

import torch

from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import Sample


def dynamic_shape_boundary_samples(sample: Sample, graph_spec: GraphSpec) -> list[Sample]:
    """Create min/max inputs for shape checks without promising valid tensor values."""
    if not graph_spec.input_spec.detected_dynamic_axis():
        return []
    return [
        resize_sample_to_shape(sample, graph_spec, "min_shape"),
        resize_sample_to_shape(sample, graph_spec, "max_shape"),
    ]


def resize_sample_to_shape(sample: Sample, graph_spec: GraphSpec, boundary: str) -> Sample:
    """Resize tensor inputs by slicing or repeating while preserving the call layout.

    Tensor values may be numerically invalid. Use these inputs for shape checks
    and timing dense computation, rather than numerical accuracy validation.
    """
    if boundary not in ("min_shape", "max_shape"):
        raise ValueError(f"Expected min_shape or max_shape, got {boundary!r}")
    args, kwargs = deepcopy(sample)
    call_layout = graph_spec.forward_signature.capture_call_layout(args, kwargs)
    forward_inputs = graph_spec.forward_signature.normalize(args, kwargs)
    for locator, tensor_spec in graph_spec.input_spec.tensor_data:
        tensor = locator.get_value(forward_inputs.arguments)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected a tensor at {locator.display_path} while constructing {boundary} inputs")
        resized = _resize_tensor(tensor, getattr(tensor_spec, boundary))
        updated_arguments = locator.set_value(forward_inputs.arguments, resized)
        if not isinstance(updated_arguments, dict):
            raise TypeError("Resizing normalized forward inputs must preserve their argument dictionary")
        forward_inputs.arguments = updated_arguments
    return call_layout.restore(forward_inputs.arguments)


def _resize_tensor(tensor: torch.Tensor, target_shape: list[int]) -> torch.Tensor:
    for axis, target_size in enumerate(target_shape):
        current_size = tensor.shape[axis]
        if current_size == target_size:
            continue
        slices = [slice(None)] * tensor.ndim
        if current_size > target_size:
            slices[axis] = slice(target_size)
            tensor = tensor[tuple(slices)]
            continue
        slices[axis] = slice(1)
        tensor = tensor[tuple(slices)]
        repeats = [1] * tensor.ndim
        repeats[axis] = target_size
        tensor = tensor.repeat(repeats)
    return tensor
