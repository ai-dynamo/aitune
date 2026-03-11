# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Global context for torch."""

BATCH_SIZE_KEY = "batch_size"
LIBRARY_LOGGING_KEY = "library_logging"


class GlobalContext:
    """Global context holding key value pairs.

    Creates only one instance (singleton) and can be used as a context manager to push and pop context.

    Example:
        >>> global_context.set("bs", 1)
        >>> with global_context:
        ...     global_context.set("bs", 2)
        ...     with global_context:
        ...         global_context.set("bs", 3)
        ...         assert global_context.get("bs") == 3
        ...     assert global_context.get("bs") == 2
        >>> assert global_context.get("bs") == 1
    """

    def __init__(self):
        """Initialize the global context."""
        self._data_stack = [{}]  # Stack to support nested contexts

    def __enter__(self):
        """Enter a new context."""
        self._data_stack.append(self._data_stack[-1].copy())  # Push a copy of the current context
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the current context."""
        self._data_stack.pop()  # Pop the current context
        return False

    @property
    def _current_data(self):
        """Return the current context."""
        return self._data_stack[-1]

    def set(self, key, value):
        """Set a key-value pair in the current context."""
        self._current_data[key] = value

    def get(self, key, default=None):
        """Get a value from the current context."""
        for data in reversed(self._data_stack):
            if key in data:
                return data[key]
        return default

    def __getitem__(self, key):
        """Get a value from the current context."""
        for data in reversed(self._data_stack):
            if key in data:
                return data[key]
        raise KeyError(key)

    def __setitem__(self, key, value):
        """Set a key-value pair in the current context."""
        self._current_data[key] = value

    def clear(self):
        """Clear the current context."""
        self._data_stack = [{}]


global_context = GlobalContext()
