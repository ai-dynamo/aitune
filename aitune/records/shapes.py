# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frontend-neutral tensor contracts and tuning metadata."""

from dataclasses import dataclass

from aitune.records.dtypes import DType


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Describe one tensor in an executable interface.

    Integer dimensions are fixed, strings are symbolic dimensions, and ``None``
    is an unnamed dynamic dimension. The specification contains only information
    declared by the executable; concrete bounds belong to :class:`TunedTensorSpec`.

    Args:
        name: Tensor name used by the executable.
        dtype: Tensor element type.
        shape: Static and dynamic dimensions declared by the executable.
    """

    name: str
    dtype: DType
    shape: tuple[int | str | None, ...]

    def __post_init__(self) -> None:
        """Validate the tensor name and declared dimensions."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("TensorSpec.name must be a non-empty string")

        if not isinstance(self.shape, tuple):
            raise ValueError(f"TensorSpec.shape must be a tuple, got {self.shape!r}")

        for index, dimension in enumerate(self.shape):
            if isinstance(dimension, bool):
                raise ValueError(
                    f"TensorSpec.shape[{index}] must be a positive integer, symbolic name, or None, got {dimension!r}"
                )
            if isinstance(dimension, int):
                if dimension <= 0:
                    raise ValueError(f"TensorSpec.shape[{index}] must be positive, got {dimension!r}")
            elif isinstance(dimension, str):
                if not dimension:
                    raise ValueError(f"TensorSpec.shape[{index}] must not be an empty symbolic name")
            elif dimension is not None:
                raise ValueError(
                    f"TensorSpec.shape[{index}] must be a positive integer, symbolic name, or None, got {dimension!r}"
                )


@dataclass(frozen=True, slots=True)
class TunedTensorSpec(TensorSpec):
    """Describe one tensor over the shape range AITune selected for an artifact.

    ``batch_axis`` is ``None`` when tuning did not identify a logical request
    batch axis.

    Args:
        name: Tensor name used by the executable.
        dtype: Tensor element type.
        shape: Static and dynamic dimensions declared by the executable.
        min_shape: Smallest shape AITune selected for the artifact.
        max_shape: Largest shape AITune selected for the artifact.
        batch_axis: Logical batch-axis index, or ``None`` when unknown.
    """

    min_shape: tuple[int, ...]
    max_shape: tuple[int, ...]
    batch_axis: int | None = None

    def __post_init__(self) -> None:
        """Validate concrete bounds against the static specification."""
        TensorSpec.__post_init__(self)
        rank = len(self.shape)
        if len(self.min_shape) != rank or len(self.max_shape) != rank:
            raise ValueError(
                f"Tuned tensor specification for {self.name!r} must have rank {rank}, "
                f"got min_shape rank {len(self.min_shape)} and max_shape rank {len(self.max_shape)}"
            )

        for index, (declared, minimum, maximum) in enumerate(
            zip(self.shape, self.min_shape, self.max_shape, strict=True)
        ):
            for label, value in (("minimum", minimum), ("maximum", maximum)):
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise ValueError(f"{label} size for axis {index} must be a positive integer, got {value!r}")
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

    @classmethod
    def from_spec(
        cls,
        spec: TensorSpec,
        *,
        min_shape: tuple[int, ...],
        max_shape: tuple[int, ...],
        batch_axis: int | None = None,
    ) -> "TunedTensorSpec":
        """Add tuning results to an existing tensor specification.

        Args:
            spec: Static tensor specification.
            min_shape: Smallest shape AITune selected for the artifact.
            max_shape: Largest shape AITune selected for the artifact.
            batch_axis: Logical batch-axis index, or ``None`` when unknown.

        Returns:
            A tuned tensor specification.
        """
        return cls(
            name=spec.name,
            dtype=spec.dtype,
            shape=spec.shape,
            min_shape=min_shape,
            max_shape=max_shape,
            batch_axis=batch_axis,
        )

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
