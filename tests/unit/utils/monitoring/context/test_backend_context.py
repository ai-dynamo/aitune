# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for with_backend_context decorator."""

from aitune.global_context import BACKEND_CONTEXT_KEY, global_context
from aitune.utils.monitoring.context.backend_context import with_backend_context


@with_backend_context
class _MockBackend:
    """Mock backend class for testing the backend_context decorator."""

    def __init__(self, key: str):
        self._key = key
        self.context_during_call = None

    def key(self) -> str:
        return self._key

    def describe(self) -> str:
        return self._key

    def public_method(self):
        """Public method that should be wrapped with backend context."""
        self.context_during_call = global_context.get(BACKEND_CONTEXT_KEY)

    def _private_method(self):
        """Private method that should NOT be wrapped."""
        self.context_during_call = global_context.get(BACKEND_CONTEXT_KEY)


def test_backend_context_decorator_wraps_public_methods():
    """Test that @backend_context decorator wraps public methods."""
    global_context.clear()
    backend = _MockBackend("test_key")

    backend.public_method()
    assert backend.context_during_call == "test_key"
    assert global_context.get(BACKEND_CONTEXT_KEY) is None


def test_backend_context_decorator_skips_private_methods():
    """Test that @backend_context decorator does not wrap private methods."""
    global_context.clear()
    backend = _MockBackend("test_key")

    backend._private_method()
    assert backend.context_during_call is None
