# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact sample metadata for matching tensor shapes."""

from typing import Any

import torch

from aitune.torch.module.locator import Locator
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.module.tensor_spec import TensorSpec


class ExactSampleMetadata(SampleMetadata):
    """Contrary to SampleMetadata, ExactSampleMetadata requires exact tensor shapes to match.

    This is useful for recording inputs and bucketing data by tensor shapes. This class has only one factory function: from_inputs
    and defines equality and hash based on locator and tensor shapes.
    """

    def __eq__(self, __value: object) -> bool:
        """Equality operator."""
        if not isinstance(__value, type(self)):
            return False

        tensor_shapes = {locator: tuple(tensor_spec.shape) for locator, tensor_spec in self._tensor_data}
        other_tensor_shapes = {locator: tuple(tensor_spec.shape) for locator, tensor_spec in __value._tensor_data}
        return tensor_shapes == other_tensor_shapes and dict(self._other_data) == dict(__value._other_data)

    def __hash__(self) -> int:
        """Compute hash of sample metadata."""
        tensor_shapes = frozenset((locator, tuple(tensor_spec.shape)) for locator, tensor_spec in self._tensor_data)
        return hash(tensor_shapes) ^ self._hash_other_data()

    @staticmethod
    def from_inputs(inputs: dict[str, Any]) -> "ExactSampleMetadata":
        """Create ExactSampleMetadata from inputs keyed by forward parameter name."""
        tensor_data, other_data = [], []
        for locator, value in Locator.find_leaves(inputs):
            if torch.is_tensor(value):
                tensor_data.append((locator, TensorSpec.from_tensor(value, batch_size=float("nan"))))
            else:
                other_data.append((locator, value))

        return ExactSampleMetadata(tuple(tensor_data), tuple(other_data), True)
