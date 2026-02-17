# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test the GlobalContext class."""

import pytest

from aitune.global_context import global_context


def test_basic_set_get():
    """Test basic set and get operations."""
    global_context.clear()  # Start with clean state

    global_context.set("test_key", "test_value")
    assert global_context.get("test_key") == "test_value"
    assert global_context["test_key"] == "test_value"

    # Test default value
    assert global_context.get("non_existent") is None
    assert global_context.get("non_existent", "default") == "default"


def test_context_manager():
    """Test context manager functionality."""
    global_context.clear()  # Start with clean state

    global_context.set("key", "outer_value")

    with global_context:
        global_context.set("key", "inner_value")
        assert global_context.get("key") == "inner_value"

    assert global_context.get("key") == "outer_value"


def test_nested_contexts():
    """Test nested context behavior."""
    global_context.clear()  # Start with clean state

    global_context.set("key", "level0")

    with global_context:
        global_context.set("key", "level1")
        assert global_context.get("key") == "level1"

        with global_context:
            global_context.set("key", "level2")
            assert global_context.get("key") == "level2"

        assert global_context.get("key") == "level1"

    assert global_context.get("key") == "level0"


def test_key_error():
    """Test KeyError is raised when accessing non-existent key with [] operator."""
    global_context.clear()  # Start with clean state

    with pytest.raises(KeyError):
        _ = global_context["non_existent_key"]


def test_clear():
    """Test clear method resets the context."""
    global_context.clear()  # Start with clean state

    global_context.set("key1", "value1")
    global_context.set("key2", "value2")

    global_context.clear()
    assert global_context.get("key1") is None
    assert global_context.get("key2") is None
