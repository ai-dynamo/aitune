# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness checking utilities."""

import copy
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_metadata import SampleMetadata


class CorrectnessValueError(ValueError):
    """Error raised when value is not finite i.e. NaN or infinity."""


class CorrectnessTensorShapeError(ValueError):
    """Error raised when tensor shapes do not match."""


class CorrectnessDynamicShapeError(ValueError):
    """Error raised when dynamic-shape boundary inference fails."""


def check_dynamic_shape_boundary_inference(
    sample: tuple[tuple, dict],
    graph_spec: GraphSpec,
    infer: Callable[..., Any],
    name: str,
):
    """Check dynamic-shape boundary inference.

    Args:
        sample: Sample inputs used as a template for min and max shape checks.
        graph_spec: Recorded graph metadata used to construct and validate boundary inputs.
        infer: Inference callable to validate.
        name: Name of the module or backend being checked.

    Raises:
        CorrectnessDynamicShapeError: If inference fails on min or max shape inputs.
        CorrectnessTensorShapeError: If inferred output tensor shapes do not match the expected metadata.

    Returns:
        None.
    """
    for boundary_name, (args, kwargs) in zip(
        ("min", "max"), _dynamic_shape_boundary_samples(sample, graph_spec), strict=False
    ):
        try:
            with torch.no_grad():
                outputs = infer(*copy.deepcopy(args), **copy.deepcopy(kwargs))
        except Exception as exc:
            raise CorrectnessDynamicShapeError(
                f"Dynamic shape correctness check failed with {boundary_name} input shapes for {name}: "
                f"{exc.__class__.__name__}: {exc}"
            ) from exc
        outputs_metadata = SampleMetadata.from_outputs(outputs)
        _check_output_tensor_shapes(graph_spec.output_spec, outputs_metadata)


def check_inference_output_correctness(
    data: list[tuple[tuple, dict]],
    output_spec: SampleMetadata,
    infer: Callable[..., Any],
    name: str,
):
    """Check recorded-sample inference output correctness.

    Args:
        data: Recorded samples to run through inference.
        output_spec: Recorded output metadata with expected output tensor shapes.
        infer: Inference callable to validate.
        name: Name of the module or backend being checked.

    Raises:
        CorrectnessValueError: If any output value is NaN or infinite.
        CorrectnessTensorShapeError: If inferred output tensor shapes do not match the expected metadata.

    Returns:
        None.
    """
    for args, kwargs in data:
        with torch.no_grad():
            outputs = infer(*copy.deepcopy(args), **copy.deepcopy(kwargs))
        _check_output_correctness(outputs, name=f"{name}.output")
        outputs_metadata = SampleMetadata.from_outputs(outputs)
        _check_output_tensor_shapes(output_spec, outputs_metadata)


def _dynamic_shape_boundary_samples(sample: tuple[tuple, dict], graph_spec: GraphSpec) -> list[tuple[tuple, dict]]:
    """Create min/max input samples for dynamic-shape correctness checks.

    Note: These samples are intended only for validating min and max shapes. Their tensor values may be
    numerically invalid and can produce NaN or Inf outputs.
    """
    if not graph_spec.input_spec.detected_dynamic_axis():
        return []

    return [
        _resize_sample_to_shape(sample, graph_spec, shape_attr="min_shape"),
        _resize_sample_to_shape(sample, graph_spec, shape_attr="max_shape"),
    ]


def _check_output_correctness(output: Any, name: str = "output", depth: int = 0):  # noqa: C901
    """Check if model outputs contain any NaN or infinity values.

    Args:
        output: Model outputs. Dict, list, tensors or scalars. Strings are ignored.
        name: Name of the output. If not provided, the name will be inferred from the output type.
        depth: Depth of the output. Auxiliary variable for recursive calls.

    Note:
        Numpy arrays are converted to torch tensors.
        Objects raise an error.

    Raises:
        CorrectnessCheckError: If any output contains NaN or infinity values
        ValueError: If output does not contain tensors or scalars
    """
    if output is None:
        return

    if isinstance(output, int | float):
        if not math.isfinite(output):
            raise CorrectnessValueError(f"Output(int, float) {name} is not finite")
        return

    if isinstance(output, str):
        return

    if isinstance(output, Sequence):
        for i, value in enumerate(output):
            _check_output_correctness(value, f"{name}[{i}]", depth + 1)
        return

    if isinstance(output, dict):
        for key, value in output.items():
            _check_output_correctness(value, f"{name}['{key}']", depth + 1)
        return

    if isinstance(output, np.ndarray):
        output = torch.from_numpy(output)

    if isinstance(output, torch.Tensor):
        if torch.isinf(output).any():
            raise CorrectnessValueError(f"Output tensor {name} contains infinity values")

        if torch.isnan(output).any():
            raise CorrectnessValueError(f"Output tensor {name} contains NaN values")


def _check_output_tensor_shapes(expected: SampleMetadata, actual: SampleMetadata):
    """Check if the output tensor shapes are the same as the original output tensor shapes."""
    errors = []
    expected_tensors = dict(expected.tensor_data)
    actual_tensors = dict(actual.tensor_data)
    if len(expected_tensors) != len(actual_tensors):
        errors.append(f"Expected {len(expected_tensors)} output tensor(s) but got {len(actual_tensors)}")

    for locator, expected_tensor in expected_tensors.items():
        actual_tensor = actual_tensors.get(locator)
        if actual_tensor is None:
            errors.append(f"Expected output tensor {locator.display_path} was not returned")
        elif not expected_tensor.matches(actual_tensor):
            errors.append(
                f"Expected tensor {locator.display_path} to have shape {expected_tensor.shape} but got {actual_tensor.shape}"
            )

    if errors:
        raise CorrectnessTensorShapeError(
            f"{len(errors)} error(s) related to output tensor shapes:\n- " + "\n- ".join(errors)
        )


def _resize_sample_to_shape(
    sample: tuple[tuple, dict],
    graph_spec: GraphSpec,
    *,
    shape_attr: str,
) -> tuple[tuple, dict]:
    """Resize each tensor input in a sample to the requested TensorSpec shape attribute.

    Note: These samples are intended only for validating min and max shapes. Their tensor values may be
    numerically invalid and can produce NaN or Inf outputs.
    """
    args, kwargs = copy.deepcopy(sample)
    forward_inputs = graph_spec.forward_signature.normalize(args, kwargs)
    for locator, tensor_spec in graph_spec.input_spec.tensor_data:
        target_shape = getattr(tensor_spec, shape_attr)
        forward_inputs.arguments = locator.set_value(
            forward_inputs.arguments,
            _resize_tensor(locator.get_value(forward_inputs.arguments), target_shape),
        )
    return forward_inputs.args, forward_inputs.kwargs


def _resize_tensor(tensor: torch.Tensor, target_shape: list[int]) -> torch.Tensor:
    """Resize a tensor by slicing or repeating each axis to match a target shape.

    Note: These samples are intended only for validating min and max shapes. Their tensor values may be
    numerically invalid and can produce NaN or Inf outputs.
    """
    for axis, target_size in enumerate(target_shape):
        current_size = tensor.shape[axis]
        if current_size == target_size:
            continue
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
