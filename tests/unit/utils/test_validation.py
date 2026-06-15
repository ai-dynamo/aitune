# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for validation helpers."""

import pytest

from aitune.utils import validation


@pytest.mark.parametrize("value", [0, 0.5, 1])
def test_ratio_accepts_values_between_zero_and_one(value: float):
    assert validation.ratio(value) is None


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_ratio_rejects_values_outside_zero_to_one(value: float):
    with pytest.raises(ValueError, match="value must be between 0 and 1"):
        validation.ratio(value)


@pytest.mark.parametrize("value", [0.25, 0.5, 0.75])
def test_ratio_accepts_values_between_custom_bounds(value: float):
    assert validation.ratio(value, min_value=0.25, max_value=0.75) is None


@pytest.mark.parametrize("value", [0.24, 0.76])
def test_ratio_rejects_values_outside_custom_bounds(value: float):
    with pytest.raises(ValueError, match="value must be between 0.25 and 0.75"):
        validation.ratio(value, min_value=0.25, max_value=0.75)


@pytest.mark.parametrize("value", [0.5, 1, 10])
def test_positive_accepts_values_greater_than_zero(value: float):
    assert validation.positive(value) is None


@pytest.mark.parametrize("value", [0, -0.1, -1])
def test_positive_rejects_values_less_than_or_equal_to_zero(value: float):
    with pytest.raises(ValueError, match="value must be positive - greater than 0"):
        validation.positive(value)


@pytest.mark.parametrize("value", [0, 1, 10])
def test_non_negative_accepts_values_greater_than_or_equal_to_zero(value: float):
    assert validation.non_negative(value) is None


@pytest.mark.parametrize("value", [-0.1, -1])
def test_non_negative_rejects_values_less_than_zero(value: float):
    with pytest.raises(ValueError, match="value must not be negative - greater than or equal to 0"):
        validation.non_negative(value)
