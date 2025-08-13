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
"""Correctness checking utilities."""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


class CorrectnessCheckError(ValueError):
    """Error raised when correctness check fails."""


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
            raise CorrectnessCheckError(f"Output(int, float) {name} is not finite")
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

    if not isinstance(output, torch.Tensor):
        raise ValueError(f"Output {name} is not a tensor but {type(output)}")

    if torch.isinf(output).any():
        raise CorrectnessCheckError(f"Output tensor {name} contains infinity values")

    if torch.isnan(output).any():
        raise CorrectnessCheckError(f"Output tensor {name} contains NaN values")
