# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

import aitune.utils.monitoring.setup_hardware_metrics as setup


@pytest.fixture(autouse=True)
def reset_singleton():
    """Resets the singleton instances before and after each test."""
    setup._DEFAULT_COLLECTOR = None
    setup._DEFAULT_RECEIVER = None
    yield
    setup._DEFAULT_COLLECTOR = None
    setup._DEFAULT_RECEIVER = None


def test_get_default_collector_singleton():
    """Test that get_default_collector returns the same object."""
    with (
        patch("aitune.utils.monitoring.setup_hardware_metrics.HARDWARE_METRICS_ENABLED", True),
        patch("aitune.utils.monitoring.setup_hardware_metrics.HardwareMetricsCollector") as mock_collector_cls,
        patch("aitune.utils.monitoring.setup_hardware_metrics.ParentReceiver") as mock_receiver_cls,
        patch("aitune.utils.monitoring.setup_hardware_metrics.atexit.register") as mock_register,
    ):
        mock_collector_instance = MagicMock()
        mock_collector_cls.return_value = mock_collector_instance
        mock_receiver_instance = MagicMock()
        mock_receiver_cls.return_value = mock_receiver_instance

        # First call
        collector1 = setup.get_default_collector()
        assert collector1 == mock_collector_instance
        mock_collector_cls.assert_called_once()
        mock_receiver_cls.assert_called_once()
        mock_register.assert_called()  # can be called multiple times by the code and metrics collection

        # Second call
        collector2 = setup.get_default_collector()
        assert collector2 == collector1
        # Should not be called again
        mock_collector_cls.assert_called_once()
        mock_receiver_cls.assert_called_once()


def test_get_default_collector_disabled():
    """Test get_default_collector when HARDWARE_METRICS_ENABLED is False."""
    with (
        patch("aitune.utils.monitoring.setup_hardware_metrics.HARDWARE_METRICS_ENABLED", False),
        patch("aitune.utils.monitoring.setup_hardware_metrics.NoOpHardwareMetricsCollector") as mock_base_collector_cls,
    ):
        mock_base_collector_instance = MagicMock()
        mock_base_collector_cls.return_value = mock_base_collector_instance

        collector = setup.get_default_collector()
        assert collector == mock_base_collector_instance
        mock_base_collector_cls.assert_called_once()


def test_get_metrics():
    """Test that get_metrics calls the receiver's get_metrics."""
    mock_receiver = MagicMock()
    setup._DEFAULT_RECEIVER = mock_receiver
    mock_df = MagicMock()
    mock_receiver.get_metrics.return_value = mock_df

    metrics = setup.get_hardware_metrics()
    assert metrics == mock_df
    mock_receiver.get_metrics.assert_called_once()


def test_dump_metrics_uses_renamed_backend_column(caplog, tmp_path, monkeypatch):
    """Test dump_metrics works correctly after get_metrics_summary renames index to Module/Backend."""
    import logging

    import pandas as pd

    monkeypatch.chdir(tmp_path)
    df = pd.DataFrame({
        "module_name": ["ModA"],
        "backend": ["TorchEagerBackend()"],
        "cpu_memory_used": [1024**3],
    })
    with caplog.at_level(logging.INFO):
        setup.dump_metrics(df)
    assert "Hardware metrics summary" in caplog.text


def test_get_metrics_none():
    """Test that get_metrics returns None if no receiver is set."""
    setup._DEFAULT_RECEIVER = None
    assert setup.get_hardware_metrics() is None
