# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Logging and settings helpers shared by the Torch-TensorRT backends."""

import logging
import os
import sys
from collections.abc import Generator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import fields
from typing import Any


@contextmanager
def torch_tensorrt_warnings(torch_tensorrt_module: Any) -> Generator[None, None, None]:
    """Limit Python and native Torch-TensorRT logs to warnings and errors."""
    python_logger = logging.getLogger("torch_tensorrt")
    original_level = python_logger.level
    python_logger.setLevel(logging.WARNING)

    try:
        with torch_tensorrt_module.logging.warnings():
            yield
    finally:
        python_logger.setLevel(original_level)


@contextmanager
def torch_tensorrt_errors(torch_tensorrt_module: Any) -> Generator[None, None, None]:
    """Hide very large Torch-TensorRT logs while saving or loading an artifact."""
    logger_names = ("torch_tensorrt", "torch._library.fake_class_registry")
    original_levels = {name: logging.getLogger(name).level for name in logger_names}
    for name in logger_names:
        logging.getLogger(name).setLevel(logging.ERROR)

    saved_fds: list[int] = []
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            sys.stdout.flush()
            sys.stderr.flush()
            # Native TensorRT messages bypass Python logging, so redirect the process descriptors too.
            saved_fds = [os.dup(1), os.dup(2)]
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            with redirect_stdout(sink), redirect_stderr(sink), torch_tensorrt_module.logging.errors():
                yield
    finally:
        if saved_fds:
            os.dup2(saved_fds[0], 1)
            os.dup2(saved_fds[1], 2)
            for fd in saved_fds:
                os.close(fd)
        for name, level in original_levels.items():
            logging.getLogger(name).setLevel(level)


def drop_unsupported_fields(config_class: type, config: dict[str, Any]) -> dict[str, Any]:
    """Drop settings unknown to the installed Torch-TensorRT, e.g. ``enabled_precisions`` removed in 2.15."""
    supported = {f.name for f in fields(config_class)}
    unsupported = sorted(set(config) - supported)
    if unsupported:
        logging.getLogger(__name__).warning(
            "Ignoring Torch-TensorRT settings not supported by the installed version: %s", unsupported
        )
    return {name: value for name, value in config.items() if name in supported}
