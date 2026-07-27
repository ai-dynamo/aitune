# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation helpers."""

from typing import TypeVar

Number = TypeVar("Number", int, float)


def in_range(value: Number, *, min_value: Number, max_value: Number, name: str | None = None) -> None:
    """Validate that value is within the provided inclusive range.

    Args:
        value: Numeric value to validate.
        min_value: Inclusive lower bound.
        max_value: Inclusive upper bound.
        name: Name of the value being validated.

    Raises:
        ValueError: If value is outside the inclusive range.
    """
    if not (min_value <= value <= max_value):
        raise ValueError(f"{name or 'value'} must be between {min_value:g} and {max_value:g}, got {value!r}.")


def positive(value: Number, *, name: str | None = None) -> None:
    """Validate that value is greater than 0.

    Args:
        value: Numeric value to validate.
        name: Name of the value being validated.

    Raises:
        ValueError: If value is less than or equal to 0.
    """
    if value <= 0:
        raise ValueError(f"{name or 'value'} must be positive, got {value!r}.")


def non_negative(value: Number, *, name: str | None = None) -> None:
    """Validate that value is greater than or equal to 0.

    Args:
        value: Numeric value to validate.
        name: Name of the value being validated.

    Raises:
        ValueError: If value is less than 0.
    """
    if value < 0:
        raise ValueError(f"{name or 'value'} must not be negative, got {value!r}.")
