# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for path utilities."""

import pytest

from aitune.torch.utils.path_utils import sanitize_filename


@pytest.mark.parametrize(
    "input_name,expected",
    [
        ("simple_name.txt", "simple_name.txt"),
        ("name with spaces", "name_with_spaces"),
        ("name/with📊special*chars", "name_with_special_chars"),
    ],
)
def test_sanitize_filename_special_chars(input_name, expected):
    """Test that special characters are replaced with underscores."""
    result = sanitize_filename(input_name)
    assert result == expected


def test_sanitize_filename_truncation():
    """Test that long filenames are truncated at max_bytes."""
    long_name = "a" * 100
    result = sanitize_filename(long_name, max_bytes=50)

    assert len(result.encode("utf-8")) <= 50
    assert result == "a" * 50
