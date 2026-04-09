# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from queue import Queue
from unittest.mock import MagicMock, patch

import pytest

from aitune.global_context import BACKEND_CONTEXT_KEY, MODULE_CONTEXT_KEY, global_context
from aitune.utils.monitoring.hardware_metrics.collector import HardwareMetricsCollector


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.get_metrics.return_value = {"metric1": 100}
    return provider


@pytest.fixture
def queue():
    return Queue()


@pytest.fixture
def collector(mock_provider, queue):
    return HardwareMetricsCollector([mock_provider], queue, interval=0.1)


def test_start_scope_begins_polling(collector, queue):
    with patch.object(collector, "_poll_stats") as mock_poll:
        collector.start_scope("test_scope")
        mock_poll.assert_called_once()
        assert len(collector.scopes) == 1


def test_end_scope_stops_polling(collector):
    collector.start_scope("test_scope")
    collector.end_scope()

    assert len(collector.scopes) == 0
    assert collector.timer is None


def test_nested_scopes(collector):
    with patch.object(collector, "_poll_stats"):
        collector.start_scope("outer")
        collector.start_scope("inner")
        assert len(collector.scopes) == 2

        collector.end_scope()
        assert len(collector.scopes) == 1

        collector.end_scope()
        assert len(collector.scopes) == 0


def test_end_scope_without_start_scope_raises(collector):
    with pytest.raises(RuntimeError, match="end_scope called without a matching start_scope"):
        collector.end_scope()


def test_collect_metrics(queue):
    provider1 = MagicMock()
    provider1.get_metrics.return_value = {"a": 1}
    provider2 = MagicMock()
    provider2.get_metrics.return_value = {"b": 2}

    collector = HardwareMetricsCollector([provider1, provider2], queue, interval=0.001)
    with global_context:
        global_context.set(MODULE_CONTEXT_KEY, "test_module")
        global_context.set(BACKEND_CONTEXT_KEY, "test_backend")
        collector.start_scope("test_scope")

    try:
        metrics = queue.get(timeout=0.1)
        assert metrics["a"] == 1
        assert metrics["b"] == 2
        assert metrics["scope"] == "test_scope"
        assert metrics["module_name"] == "test_module"
        assert metrics["backend"] == "test_backend"
        assert metrics["timestamp"] > 0

        # check we collect next sample
        assert queue.get(timeout=0.1) is not None
    finally:
        collector.end_scope()
