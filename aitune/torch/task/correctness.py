# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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
"""Correctness checking utilities."""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from aitune.torch.module.tensor_spec import TensorSpec


class CorrectnessValueError(ValueError):
    """Error raised when value is not finite i.e. NaN or infinity."""


class CorrectnessTensorShapeError(ValueError):
    """Error raised when tensor shapes do not match."""


def check_output_correctness(output: Any, name: str = "output", depth: int = 0):  # noqa: C901
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

    if isinstance(output, (int, float)):
        if not math.isfinite(output):
            raise CorrectnessValueError(f"Output(int, float) {name} is not finite")
        return

    if isinstance(output, str):
        return

    if isinstance(output, Sequence):
        for i, value in enumerate(output):
            check_output_correctness(value, f"{name}[{i}]", depth + 1)
        return

    if isinstance(output, dict):
        for key, value in output.items():
            check_output_correctness(value, f"{name}['{key}']", depth + 1)
        return

    if isinstance(output, np.ndarray):
        output = torch.from_numpy(output)

    if isinstance(output, torch.Tensor):
        if torch.isinf(output).any():
            raise CorrectnessValueError(f"Output tensor {name} contains infinity values")

        if torch.isnan(output).any():
            raise CorrectnessValueError(f"Output tensor {name} contains NaN values")


def check_output_tensor_shapes(expected_tensor_specs: list[TensorSpec], actual_tensor_specs: list[TensorSpec]):
    """Check if the output tensor shapes are the same as the original output tensor shapes."""
    errors = []
    for orig_tensor, tuned_tensor in zip(expected_tensor_specs, actual_tensor_specs, strict=True):
        if not orig_tensor.matches(tuned_tensor):
            errors.append(
                f"Expected tensor {orig_tensor.name} to have shape {orig_tensor.shape} but got {tuned_tensor.shape}"
            )

    if errors:
        raise CorrectnessTensorShapeError(
            f"{len(errors)} error(s) related to output tensor shapes:\n- " + "\n- ".join(errors)
        )
