# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frontend-neutral tensor contracts and tuning metadata."""

from dataclasses import dataclass

from aitune.records.dtypes import DType


def _validate_declared_shape(shape: tuple[int | str | None, ...]) -> None:
    """Validate static and symbolic dimensions."""
    if not isinstance(shape, tuple):
        raise ValueError(f"BoundedTensorSpec.shape must be a tuple, got {shape!r}")

    for index, dimension in enumerate(shape):
        if isinstance(dimension, bool):
            raise ValueError(
                f"BoundedTensorSpec.shape[{index}] must be a positive integer, symbolic name, or None, "
                f"got {dimension!r}"
            )
        if isinstance(dimension, int) and dimension <= 0:
            raise ValueError(f"BoundedTensorSpec.shape[{index}] must be positive, got {dimension!r}")
        if isinstance(dimension, str) and not dimension:
            raise ValueError(f"BoundedTensorSpec.shape[{index}] must not be an empty symbolic name")
        if not isinstance(dimension, int | str) and dimension is not None:
            raise ValueError(
                f"BoundedTensorSpec.shape[{index}] must be a positive integer, symbolic name, or None, "
                f"got {dimension!r}"
            )


def _validate_bound(name: str, label: str, shape: tuple[int, ...], rank: int) -> None:
    """Validate one concrete shape bound."""
    if not isinstance(shape, tuple):
        raise ValueError(f"BoundedTensorSpec.{label}_shape must be a tuple, got {shape!r}")
    if len(shape) != rank:
        raise ValueError(f"{label}_shape for {name!r} must have rank {rank}, got {len(shape)}")
    for index, value in enumerate(shape):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{label} size for axis {index} must be a positive integer, got {value!r}")


@dataclass(frozen=True, slots=True)
class BoundedTensorSpec:
    """Describe one tensor over the shape range established by tuning.

    Integer dimensions are fixed, strings are symbolic dimensions, and ``None``
    is an unnamed dynamic dimension. Concrete minimum and maximum shapes record
    the domain AITune validated for the artifact. ``batch_axis`` is ``None`` when
    tuning did not identify a logical request batch axis.

    Args:
        name: Tensor name used by the executable.
        dtype: Tensor element type.
        shape: Static and dynamic dimensions declared by the executable.
        min_shape: Smallest shape AITune selected for the artifact.
        max_shape: Largest shape AITune selected for the artifact.
        batch_axis: Logical batch-axis index, or ``None`` when unknown.
    """

    name: str
    dtype: DType
    shape: tuple[int | str | None, ...]
    min_shape: tuple[int, ...]
    max_shape: tuple[int, ...]
    batch_axis: int | None = None

    def __post_init__(self) -> None:
        """Validate the tensor declaration and its concrete bounds."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("BoundedTensorSpec.name must be a non-empty string")
        if not isinstance(self.dtype, DType):
            raise ValueError(f"BoundedTensorSpec.dtype must be a DType, got {self.dtype!r}")

        _validate_declared_shape(self.shape)
        rank = len(self.shape)
        _validate_bound(self.name, "minimum", self.min_shape, rank)
        _validate_bound(self.name, "maximum", self.max_shape, rank)

        for index, (declared, minimum, maximum) in enumerate(
            zip(self.shape, self.min_shape, self.max_shape, strict=True)
        ):
            if minimum > maximum:
                raise ValueError(
                    f"Axis {index} of {self.name!r} must satisfy min <= max, got min={minimum}, max={maximum}"
                )
            if isinstance(declared, int) and (minimum, maximum) != (declared, declared):
                raise ValueError(
                    f"Fixed axis {index} of {self.name!r} must remain {declared}, got min={minimum}, max={maximum}"
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
