# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for validation helpers."""

import pytest

from aitune.utils import validation


@pytest.mark.parametrize("value", [0, 0.5, 1])
def test_in_range_accepts_values_between_zero_and_one(value: float):
    assert validation.in_range(value, min_value=0, max_value=1) is None


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_in_range_rejects_values_outside_zero_to_one(value: float):
    with pytest.raises(ValueError) as error:
        validation.in_range(value, min_value=0, max_value=1, name="ratio")
    assert str(error.value) == f"ratio must be between 0 and 1, got {value!r}."


@pytest.mark.parametrize("value", [0.25, 0.5, 0.75])
def test_in_range_accepts_values_between_custom_bounds(value: float):
    assert validation.in_range(value, min_value=0.25, max_value=0.75) is None


@pytest.mark.parametrize("value", [0.24, 0.76])
def test_in_range_rejects_values_outside_custom_bounds(value: float):
    with pytest.raises(ValueError) as error:
        validation.in_range(value, min_value=0.25, max_value=0.75)
    assert str(error.value) == f"value must be between 0.25 and 0.75, got {value!r}."


@pytest.mark.parametrize("value", [0.5, 1, 10])
def test_positive_accepts_values_greater_than_zero(value: float):
    assert validation.positive(value) is None


@pytest.mark.parametrize("value", [0, -0.1, -1])
def test_positive_rejects_values_less_than_or_equal_to_zero(value: float):
    with pytest.raises(ValueError) as error:
        validation.positive(value, name="number")
    assert str(error.value) == f"number must be positive, got {value!r}."


@pytest.mark.parametrize("value", [0, 1, 10])
def test_non_negative_accepts_values_greater_than_or_equal_to_zero(value: float):
    assert validation.non_negative(value) is None


@pytest.mark.parametrize("value", [-0.1, -1])
def test_non_negative_rejects_values_less_than_zero(value: float):
    with pytest.raises(ValueError) as error:
        validation.non_negative(value, name="number")
    assert str(error.value) == f"number must not be negative, got {value!r}."
