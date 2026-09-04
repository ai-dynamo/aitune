# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Torch-TensorRT logging helpers."""

import logging

import pytest

from aitune.torch.backend.torch_tensorrt_logging import torch_tensorrt_warnings


def test_torch_tensorrt_warnings_limits_python_and_native_logging(mocker):
    torch_tensorrt = mocker.Mock()
    native_context = mocker.MagicMock()
    torch_tensorrt.logging.warnings.return_value = native_context
    python_logger = logging.getLogger("torch_tensorrt")
    original_level = python_logger.level

    with torch_tensorrt_warnings(torch_tensorrt):
        assert python_logger.level == logging.WARNING
        assert logging.getLogger("torch_tensorrt.dynamo.conversion").getEffectiveLevel() == logging.WARNING

    assert python_logger.level == original_level
    native_context.__enter__.assert_called_once_with()
    native_context.__exit__.assert_called_once()


def test_torch_tensorrt_warnings_restores_python_logging_after_failure(mocker):
    torch_tensorrt = mocker.Mock()
    torch_tensorrt.logging.warnings.return_value = mocker.MagicMock()
    python_logger = logging.getLogger("torch_tensorrt")
    original_level = python_logger.level

    with pytest.raises(RuntimeError, match="compilation failed"):
        with torch_tensorrt_warnings(torch_tensorrt):
            raise RuntimeError("compilation failed")

    assert python_logger.level == original_level
