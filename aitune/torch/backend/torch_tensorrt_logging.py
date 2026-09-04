# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Logging helpers shared by the Torch-TensorRT backends."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def torch_tensorrt_warnings(torch_tensorrt_module: Any) -> Iterator[None]:
    """Limit Python and native Torch-TensorRT logs to warnings and errors."""
    python_logger = logging.getLogger("torch_tensorrt")
    original_level = python_logger.level
    python_logger.setLevel(logging.WARNING)

    try:
        with torch_tensorrt_module.logging.warnings():
            yield
    finally:
        python_logger.setLevel(original_level)
