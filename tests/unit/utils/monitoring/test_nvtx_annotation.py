# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from aitune.global_context import BACKEND_CONTEXT_KEY, MODULE_CONTEXT_KEY, global_context
from aitune.utils.monitoring.nvtx_annotation import NVTX_ANNOTATION_DOMAIN, annotate_with_nvtx


@pytest.fixture
def mock_nvtx():
    with patch("aitune.utils.monitoring.nvtx_annotation.nvtx") as mock:
        mock_annotate = MagicMock()
        mock.annotate.return_value = mock_annotate
        yield mock


@pytest.fixture(autouse=True)
def clean_context():
    """Clean up the global context before and after each test."""
    global_context.clear()
    yield
    global_context.clear()


def test_context_manager(mock_nvtx):
    with annotate_with_nvtx(name="test_scope", color="blue"):
        pass

    mock_nvtx.annotate.assert_called_once_with(message="test_scope", color="blue", domain=NVTX_ANNOTATION_DOMAIN)
    mock_nvtx.annotate.return_value.__enter__.assert_called_once()
    mock_nvtx.annotate.return_value.__exit__.assert_called_once()


def test_context_manager_requires_name():
    with pytest.raises(ValueError, match="Name is required"):
        with annotate_with_nvtx():
            pass


def test_context_manager_calls_exit_on_exception(mock_nvtx):
    with pytest.raises(ValueError):
        with annotate_with_nvtx(name="test"):
            raise ValueError("test error")

    mock_nvtx.annotate.return_value.__enter__.assert_called_once()
    mock_nvtx.annotate.return_value.__exit__.assert_called_once()


def test_decorator(mock_nvtx):
    @annotate_with_nvtx(name="decorated_func", color="green")
    def my_func():
        return 42

    # nvtx.annotate must not be called at decoration time
    mock_nvtx.annotate.assert_not_called()

    my_func()

    mock_nvtx.annotate.assert_called_once_with(message="decorated_func", color="green", domain=NVTX_ANNOTATION_DOMAIN)
    mock_nvtx.annotate.return_value.__enter__.assert_called_once()
    mock_nvtx.annotate.return_value.__exit__.assert_called_once()


def test_decorator_uses_function_name_when_no_name_provided(mock_nvtx):
    @annotate_with_nvtx()
    def my_custom_func():
        return 42

    my_custom_func()

    mock_nvtx.annotate.assert_called_once_with(message="my_custom_func", color=None, domain=NVTX_ANNOTATION_DOMAIN)


def test_context_manager_with_module_context(mock_nvtx):
    with global_context:
        global_context.set(MODULE_CONTEXT_KEY, "TestModule")
        with annotate_with_nvtx(name="test_scope"):
            pass

    mock_nvtx.annotate.assert_called_once_with(
        message="test_scope module:TestModule", color=None, domain=NVTX_ANNOTATION_DOMAIN
    )


def test_context_manager_with_backend_context(mock_nvtx):
    with global_context:
        global_context.set(BACKEND_CONTEXT_KEY, "backend_key")
        with annotate_with_nvtx(name="test_scope"):
            pass

    mock_nvtx.annotate.assert_called_once_with(
        message="test_scope backend:backend_key", color=None, domain=NVTX_ANNOTATION_DOMAIN
    )


def test_context_manager_with_both_module_and_backend_context(mock_nvtx):
    with global_context:
        global_context.set(MODULE_CONTEXT_KEY, "TestModule")
        global_context.set(BACKEND_CONTEXT_KEY, "backend_key")
        with annotate_with_nvtx(name="test_scope"):
            pass

    mock_nvtx.annotate.assert_called_once_with(
        message="test_scope module:TestModule backend:backend_key", color=None, domain=NVTX_ANNOTATION_DOMAIN
    )
