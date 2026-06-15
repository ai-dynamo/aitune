# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation helpers."""

from typing import TypeVar

Number = TypeVar("Number", int, float)


def ratio(value: Number, *, min_value: Number = 0, max_value: Number = 1) -> None:
    """Validate that value is within the provided inclusive range.

    Args:
        value: Numeric value to validate.
        min_value: Inclusive lower bound.
        max_value: Inclusive upper bound.

    Raises:
        ValueError: If value is outside the inclusive range.
    """
    if not (min_value <= value <= max_value):
        raise ValueError(f"value must be between {min_value:g} and {max_value:g}.")


def positive(value: Number) -> None:
    """Validate that value is greater than 0.

    Args:
        value: Numeric value to validate.

    Raises:
        ValueError: If value is less than or equal to 0.
    """
    if value <= 0:
        raise ValueError("value must be positive - greater than 0.")


def non_negative(value: Number) -> None:
    """Validate that value is greater than or equal to 0.

    Args:
        value: Numeric value to validate.

    Raises:
        ValueError: If value is less than 0.
    """
    if value < 0:
        raise ValueError("value must not be negative - greater than or equal to 0.")
