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
"""Contains TensorSpec which represents metadata of a tensor."""

import math
from dataclasses import asdict, dataclass
from enum import Enum, auto
from typing import Any

import torch


class InfoLevel(Enum):
    """Enum representing different levels of information detail."""

    SHORT = auto()
    MEDIUM = auto()
    FULL = auto()


@dataclass
class TensorSpec:
    """TensorSpec is used to describe tensor metadata.

    Attributes:
        name (str) - symbolic name of the tensor
        shape (list[Union[str, int]]) - shape of the tensor, int is a real dimension, str is a symbolic dimension
        min_shape (list[int]) - minimum dimensions seen so far
        max_shape (list[int]) - maximum dimensions seen so far
        dtype (torch.dtype) - dtype of the tensor
        _bs_multipliers (list[float]) - batch size multipliers for each axis
    """

    name: str
    shape: list[str | int]
    min_shape: list[int]
    max_shape: list[int]
    dtype: torch.dtype | None
    _bs_multipliers: list[float]

    @staticmethod
    def from_tensor(name: str, tensor: torch.Tensor, batch_size: int):
        """Create TensorSpec from tensor.

        Args:
            name: Name of the tensor
            tensor: Tensor to create TensorSpec from
            batch_size: Batch size
        """
        shape = list(tensor.shape)
        self = TensorSpec(
            name=name,
            shape=shape,
            min_shape=shape[:],
            max_shape=shape[:],
            dtype=tensor.dtype,
            _bs_multipliers=[size / batch_size for size in shape],
        )
        return self

    def __repr__(self) -> str:
        """Get representation of TensorSpec."""
        return self.describe(InfoLevel.MEDIUM)

    def __str__(self) -> str:
        """Get string representation of TensorSpec."""
        return self.describe(InfoLevel.SHORT)

    def __eq__(self, other: object) -> bool:
        """Check if two TensorSpec are equal.

        Tensors of the same name and rank are considered equal.
        Particular dimensions sizes can be different as there can be dynamic dimensions.
        """
        if not isinstance(other, TensorSpec):
            return False
        return self.name == other.name and len(self.shape) == len(other.shape)

    def __hash__(self) -> int:
        """Hash of TensorSpec.

        Tensors of the same name and rank are considered equal.
        Particular dimensions sizes can be different as there can be dynamic dimensions.
        """
        return hash((self.name, len(self.shape)))

    def describe(self, info_level: InfoLevel = InfoLevel.FULL) -> str:
        """Get information describing TensorSpec."""
        if info_level == InfoLevel.SHORT:
            return self.name
        elif info_level == InfoLevel.MEDIUM:
            shapes = ", ".join(str(dim) for dim in self.shape)
            return f"{self.name}[{shapes}]"
        elif info_level == InfoLevel.FULL:
            shapes = ", ".join(str(dim) for dim in self.shape)
            min_shape = ", ".join(str(dim) for dim in self.min_shape)
            max_shape = ", ".join(str(dim) for dim in self.max_shape)
            return f"{self.name}[{shapes}] min_shape=[{min_shape}] max_shape=[{max_shape}] dtype={self.dtype}"  # type: ignore[bad-return-type]

    def update_shapes_seen(self, other: "TensorSpec"):
        """Update shapes seen from other TensorSpec.

        Tensor have to have same rank in order to update self.

        The algorithm for detecting batch dimension is the following: given two different batch sizes, if batch size
        multiplier is the same for same axis in both tensors and is an integer, then it is a batch dimension otherwise
        it is a dynamic dimension.

        The reason for using multipliers is that some models stack input tensor vertically and the resulting input has
        double of the batch size i.e. local batch size is 2x the global batch size.

        Example of the algorithm - let's assume we observed tensor[1, 2, 3, 4] given bs=1. We calculated multipliers
        to be [1, 2, 3, 4]. Now if we see tensor[2, 8, 6, 4] with bs=2, we calculate multipliers to be
        [1, 4, 3, 2]. We can make some conclusions:
        - 0th axis is batch axis, multiplier is 1
        - 1st axis is dynamic axis, multiplier is 2 and 4 - this could be for example length in LLMs
        - 2nd axis is batch axis, multiplier is 3 - the input tensor is v-stacked thus multiplier is 3
        - 3rd axis is static axis - never changes w.r.t. batch size

        This algorithm is not foolproof and can fail in some cases e.g. sequence length in LLMs is equal to batch size
        which is unlikely to happen in practice. However to mitigate this risk, there is additional check for batch axis
        multiplier which has to be an integer.
        """
        if len(self.shape) != len(other.shape):
            raise ValueError("Tensors must have the same rank")
        for i in range(len(self.shape)):
            if self.shape[i] != other.shape[i]:
                self.min_shape[i] = min(self.min_shape[i], other.min_shape[i])
                self.max_shape[i] = max(self.max_shape[i], other.max_shape[i])
                if self._bs_multipliers[i] == other._bs_multipliers[i] and TensorSpec.is_int(self._bs_multipliers[i]):
                    self.shape[i] = f"batch{i}"
                else:
                    self._bs_multipliers[i] = float("nan")
                    self.shape[i] = f"dim{i}"

    def to_dict(self) -> dict[str, Any]:
        """Convert TensorSpec to a serializable dictionary."""
        return {"type": self.__class__.__name__} | asdict(self)

    def get_batch_axis_multipliers(self) -> dict[int, int]:
        """Return mapping for batch axis and its multiplier."""
        result = {}
        for i, shape in enumerate(self.shape):
            if isinstance(shape, str) and shape.startswith("batch"):
                result[i] = int(self._bs_multipliers[i])
        return result

    def get_max_batch_size(self) -> int:
        """Get max batch size from tensor spec."""
        max_batch_size = 1
        for axis in self.get_batch_axis_multipliers().keys():
            max_batch_size = max(max_batch_size, self.max_shape[axis])
        return max_batch_size

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TensorSpec":
        """Create TensorSpec from dictionary."""
        if data.get("type") != TensorSpec.__name__:
            raise ValueError(f"Invalid dictionary format for {TensorSpec.__name__}")
        del data["type"]
        return TensorSpec(**data)

    @staticmethod
    def is_int(value: float) -> bool:
        """Check if value is an integer by checking value not the type."""
        return not math.isnan(value) and int(value) == value

    def has_batch_axis(self) -> bool:
        """Check if tensor has batch axis."""
        return any(isinstance(dim, str) and dim.startswith("batch") for dim in self.shape)
