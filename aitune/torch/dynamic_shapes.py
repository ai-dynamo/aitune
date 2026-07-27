# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic shape dimension definitions."""

from dataclasses import asdict, dataclass
from typing import Any

from aitune.exceptions import AITuneUserInputError
from aitune.torch.module.forward_signature import ForwardInputPath, validate_forward_input_path
from aitune.utils import validation

__all__ = ["BatchDim", "DynamicDim"]


@dataclass(frozen=True, slots=True)
class DynamicDim:
    """Named dynamic dimension with inclusive bounds."""

    name: str
    min: int
    max: int
    opt: int | None = None

    def __post_init__(self) -> None:
        """Validate the dimension name and bounds."""
        if not isinstance(self.name, str) or not self.name:
            raise AITuneUserInputError(f"Dynamic dimension name must be a non-empty string, got {self.name!r}.")
        for field_name, value in (("min", self.min), ("max", self.max), ("opt", self.opt)):
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise AITuneUserInputError(
                    f"{field_name} for dynamic dimension {self.name!r} must be an integer, got {value!r}."
                )
            validation.positive(value, name=f"{field_name} for dynamic dimension {self.name!r}")
        if not self.min < self.max:
            raise AITuneUserInputError(
                f"Dynamic dimension {self.name!r} must satisfy min < max, got min={self.min}, max={self.max}."
            )
        if self.opt is not None and not self.min <= self.opt <= self.max:
            raise AITuneUserInputError(
                f"Dynamic dimension {self.name!r} must satisfy min <= opt <= max, got "
                f"min={self.min}, opt={self.opt}, max={self.max}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert the dynamic dimension to a serializable dictionary."""
        return {"type": self.__class__.__name__} | asdict(self)


@dataclass(frozen=True, slots=True)
class BatchDim(DynamicDim):
    """Dynamic dimension representing the module's logical batch size."""


ShapeDefinition = tuple[int | DynamicDim, ...]
DynamicShapes = dict[ForwardInputPath, ShapeDefinition]


def dynamic_shapes_to_json(dynamic_shapes: DynamicShapes | None) -> list[dict[str, Any]] | None:
    """Convert dynamic shape definitions to JSON-compatible data.

    >>> shapes = {("options", "mask"): (BatchDim("batch", min=1, opt=2, max=4), 128)}
    >>> dynamic_shapes_to_json(shapes)  # doctest: +NORMALIZE_WHITESPACE
    [{'path': ['options', 'mask'],
      'shape': [{'type': 'BatchDim', 'name': 'batch', 'min': 1, 'max': 4, 'opt': 2}, 128]}]
    """
    if dynamic_shapes is None:
        return None

    return [
        {
            "path": list(path) if isinstance(path, tuple) else path,
            "shape": [dimension if isinstance(dimension, int) else dimension.to_dict() for dimension in shape],
        }
        for path, shape in dynamic_shapes.items()
    ]


def validate_dynamic_shape_definitions(dynamic_shapes: object) -> None:
    """Validate explicit shape definitions before recording.

    Each mapping value must define the complete input shape using non-negative static sizes or dynamic dimensions.
    Repeated dynamic dimension names define shared dimensions and must use identical definitions.
    """
    if not isinstance(dynamic_shapes, dict):
        raise AITuneUserInputError(f"dynamic_shapes must be a dictionary, got {dynamic_shapes!r}.")

    dimension_definitions_by_name: dict[str, DynamicDim] = {}
    for path, shape_definition in dynamic_shapes.items():
        validate_forward_input_path(path)
        if not isinstance(shape_definition, tuple):
            raise AITuneUserInputError(f"Dynamic shape for input {path!r} must be a tuple, got {shape_definition!r}.")

        for axis, axis_definition in enumerate(shape_definition):
            # Integer axis definitions are static sizes.
            if isinstance(axis_definition, int) and not isinstance(axis_definition, bool):
                validation.non_negative(axis_definition, name=f"Axis {axis} for input {path!r}")
                continue
            if not isinstance(axis_definition, DynamicDim):
                raise AITuneUserInputError(
                    f"Axis {axis} for input {path!r} must be a static integer or dynamic dimension, "
                    f"got {axis_definition!r}."
                )

            # Repeated names identify shared dimensions whose definitions must match.
            if axis_definition.name not in dimension_definitions_by_name:
                dimension_definitions_by_name[axis_definition.name] = axis_definition
            else:
                expected_definition = dimension_definitions_by_name[axis_definition.name]
                if type(expected_definition) is not type(axis_definition) or (
                    expected_definition.min,
                    expected_definition.opt,
                    expected_definition.max,
                ) != (axis_definition.min, axis_definition.opt, axis_definition.max):
                    raise AITuneUserInputError(
                        f"Dynamic dimension {axis_definition.name!r} has conflicting definitions: "
                        f"{expected_definition!r} and {axis_definition!r}."
                    )
