# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for CUDA utilities."""

import pytest
import torch

from aitune.torch.utils.cuda_utils import assert_is_available, is_available, synchronize

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
