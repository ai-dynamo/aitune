# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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
"""Test for CUDA utilities."""

import pytest
import torch

from aitune.torch.utils.cuda import assert_is_available, is_available, synchronize

requires_cuda = pytest.mark.skipif(not is_available(), reason="CUDA is not available")


def test_is_available():
    """Test the is_available function."""
    assert is_available() == torch.cuda.is_available()


def test_synchronize(mocker):
    """Test the synchronize function."""
    mocker.patch("torch.cuda.synchronize")
    mocker.patch("torch.cuda.is_available", return_value=is_available())

    synchronize()

    torch.cuda.is_available.assert_called_once()
    if is_available():
        torch.cuda.synchronize.assert_called_once()


def test_assert_is_available():
    """Test the synchronize function."""
    if is_available():
        assert_is_available()
    else:
        with pytest.raises(RuntimeError):
            assert_is_available()
