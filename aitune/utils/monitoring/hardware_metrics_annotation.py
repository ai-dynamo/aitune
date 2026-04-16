# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Annotation for collecting hardware metrics."""

from functools import wraps

from aitune.utils.monitoring.setup_hardware_metrics import get_default_collector


class collect_hardware_metrics:  # noqa: N801
    """Decorator/context manager for collecting hardware metrics."""

    def __init__(self, name: str | None = None):
        """Initializes the hardware metrics collector."""
        self.name = name

    def __enter__(self):
        """Enters the context manager."""
        if self.name is None:
            raise ValueError("name must be provided when using collect_hardware_metrics as a context manager")
        get_default_collector().start_scope(self.name)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exits the context manager."""
        get_default_collector().end_scope()
        return False

    def __call__(self, func):
        """Wraps the function as a decorator."""
        name = self.name if self.name is not None else func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            get_default_collector().start_scope(name)
            try:
                result = func(*args, **kwargs)
            finally:
                get_default_collector().end_scope()
            return result

        return wrapper
