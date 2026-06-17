# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for collecting JIT tuning-data reports from functional tests."""

import os
import tempfile
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import TypeVar

from aitune.torch.tune_data.reporting import snapshot_tuning_data

AITUNE_OUTPUT_DIR_ENV = "AITUNE_OUTPUT_DIR"

R = TypeVar("R")


def tuning_data_path(test_file: str | Path, output_dir: Path | None = None) -> Path:
    """Return the tuning-data report path for a functional test file."""
    root = output_dir or os.environ.get(AITUNE_OUTPUT_DIR_ENV) or tempfile.gettempdir()
    return Path(root) / f"{Path(test_file).stem}.json"


def flush_tuning_data(
    test_file: str | Path,
    output_dir: Path | None = None,
) -> Path | None:
    """Flush the active JIT tuning report to the functional test's artifact file."""
    report_path = tuning_data_path(test_file, output_dir=output_dir)
    return snapshot_tuning_data(report_path)


def collect_tuning_data(
    test_file: str | Path,
    output_dir: Path | None = None,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Decorate a functional test so its JIT tuning report is saved as an artifact."""

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> R:
            try:
                return func(*args, **kwargs)
            finally:
                flush_tuning_data(test_file, output_dir=output_dir)

        return wrapper

    return decorator
