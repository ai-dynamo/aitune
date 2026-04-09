# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decorator for tracking the current backend in GlobalContext."""

import functools
import inspect

from aitune.global_context import BACKEND_CONTEXT_KEY, global_context


def with_backend_context(cls):
    """Class decorator that wraps all public methods with backend context."""
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        # Skip private/magic methods and static methods
        if name.startswith("_") or name in ("key", "describe"):
            continue

        original_method = method

        @functools.wraps(original_method)
        def wrapper(self, *args, _method=original_method, **kwargs):
            with global_context:
                global_context.set(BACKEND_CONTEXT_KEY, self.describe())
                return _method(self, *args, **kwargs)

        setattr(cls, name, wrapper)

    return cls
