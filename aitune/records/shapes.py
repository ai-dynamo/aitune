# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frontend-neutral tensor contracts and tuning metadata."""

from dataclasses import dataclass

from aitune.records.dtypes import DType


def _validate_bound(field_name: str, shape: tuple[int, ...]) -> None:
    """Validate one concrete shape bound."""
    if not isinstance(shape, tuple):
        raise ValueError(f"BoundedTensorSpec.{field_name} must be a tuple, got {shape!r}")
    for index, value in enumerate(shape):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{field_name}[{index}] must be a positive integer, got {value!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class BoundedTensorSpec:
    """Describe one tensor over the shape range established by tuning.

    Minimum and maximum shapes record the concrete domain AITune validated for
    the artifact. An axis is fixed when both bounds are equal and dynamic when
    they differ. The first axis is treated as the logical batch axis by default;
    ``batch_axis=None`` explicitly describes an unbatched tensor.

    Args:
        name: Tensor name used by the executable.
        dtype: Tensor element type.
        min_shape: Smallest accepted tensor shape.
        max_shape: Largest accepted tensor shape.
        batch_axis: Logical batch-axis index, or ``None`` for an unbatched tensor.
    """

    name: str
    dtype: DType
    min_shape: tuple[int, ...]
    max_shape: tuple[int, ...]
    batch_axis: int | None = 0

    def __post_init__(self) -> None:
        """Validate the tensor declaration and its concrete bounds."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("BoundedTensorSpec.name must be a non-empty string")
        if not isinstance(self.dtype, DType):
            raise ValueError(f"BoundedTensorSpec.dtype must be a DType, got {self.dtype!r}")

        _validate_bound("min_shape", self.min_shape)
        _validate_bound("max_shape", self.max_shape)
        rank = len(self.min_shape)
        if len(self.max_shape) != rank:
            raise ValueError(f"Bounds for {self.name!r} must have the same rank, got {rank} and {len(self.max_shape)}")

        for index, (minimum, maximum) in enumerate(zip(self.min_shape, self.max_shape, strict=True)):
            if minimum > maximum:
                raise ValueError(
                    f"Axis {index} of {self.name!r} must satisfy min <= max, got min={minimum}, max={maximum}"
                )

        if self.batch_axis is not None and (
            not isinstance(self.batch_axis, int) or isinstance(self.batch_axis, bool) or not 0 <= self.batch_axis < rank
        ):
            raise ValueError(f"batch_axis must be a valid axis for {self.name!r}, got {self.batch_axis!r}")

    @property
    def min_batch_size(self) -> int | None:
        """Return the smallest batch size AITune selected, if known."""
        if self.batch_axis is None:
            return None
        return self.min_shape[self.batch_axis]

    @property
    def max_batch_size(self) -> int | None:
        """Return the largest batch size AITune selected, if known."""
        if self.batch_axis is None:
            return None
        return self.max_shape[self.batch_axis]
