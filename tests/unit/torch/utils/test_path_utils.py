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
