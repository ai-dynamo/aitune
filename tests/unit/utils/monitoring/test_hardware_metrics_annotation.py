# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from aitune.utils.monitoring.hardware_metrics_annotation import collect_hardware_metrics

_PATCH = "aitune.utils.monitoring.hardware_metrics_annotation.get_default_collector"


@pytest.fixture
def mock_collector():
    return MagicMock()


def test_context_manager(mock_collector):
    with patch(_PATCH, return_value=mock_collector):
        with collect_hardware_metrics(name="test_scope"):
            pass

    mock_collector.start_scope.assert_called_once_with("test_scope")
    mock_collector.end_scope.assert_called_once()


def test_context_manager_calls_end_scope_on_exception(mock_collector):
    with patch(_PATCH, return_value=mock_collector):
        with pytest.raises(ValueError):
            with collect_hardware_metrics(name="test"):
                raise ValueError("test error")

    mock_collector.start_scope.assert_called_once_with("test")
    mock_collector.end_scope.assert_called_once()


def test_decorator(mock_collector):
    with patch(_PATCH, return_value=mock_collector):

        @collect_hardware_metrics(name="decorated_func")
        def my_func():
            return 42

        result = my_func()

    assert result == 42
    mock_collector.start_scope.assert_called_once_with("decorated_func")
    mock_collector.end_scope.assert_called_once()


def test_context_manager_without_name_raises(mock_collector):
    with patch(_PATCH, return_value=mock_collector):
        with pytest.raises(ValueError, match="name must be provided"):
            with collect_hardware_metrics():
                pass

    mock_collector.start_scope.assert_not_called()


def test_decorator_calls_end_scope_on_exception(mock_collector):
    with patch(_PATCH, return_value=mock_collector):

        @collect_hardware_metrics()
        def my_func():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            my_func()

    mock_collector.start_scope.assert_called_once_with("my_func")
    mock_collector.end_scope.assert_called_once()
