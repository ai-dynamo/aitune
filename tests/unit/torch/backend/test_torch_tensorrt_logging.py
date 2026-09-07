# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Torch-TensorRT logging helpers."""

import logging
import os

import pytest

from aitune.torch.backend.torch_tensorrt_logging import torch_tensorrt_errors, torch_tensorrt_warnings


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


def test_torch_tensorrt_errors_suppresses_serialization_diagnostics(mocker, capfd):
    torch_tensorrt = mocker.Mock()
    native_context = mocker.MagicMock()
    torch_tensorrt.logging.errors.return_value = native_context
    logger_names = ("torch_tensorrt", "torch._library.fake_class_registry")
    original_levels = {name: logging.getLogger(name).level for name in logger_names}

    with torch_tensorrt_errors(torch_tensorrt):
        for name in logger_names:
            assert logging.getLogger(name).level == logging.ERROR
        print("python serialization diagnostic")
        os.write(1, b"native serialization diagnostic\n")

    for name, level in original_levels.items():
        assert logging.getLogger(name).level == level
    native_context.__enter__.assert_called_once_with()
    native_context.__exit__.assert_called_once()
    captured = capfd.readouterr()
    assert "serialization diagnostic" not in captured.out
