# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pandas as pd
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


# ── _split_for_logging ────────────────────────────────────────────────────────


def _make_flat(n_gpus: int) -> pd.DataFrame:
    """Build a flat DataFrame as dump_metrics would see it for *n_gpus* GPUs."""
    cols = ["Module", "Backend", "Host\nMem [GB]"]
    for i in range(n_gpus):
        cols.append(f"Cuda:{i}\nMem [GB]")
    for i in range(n_gpus):
        cols += [f"Cuda:{i}\nUtil% mean", f"Cuda:{i}\nUtil% max"]
    for _ in range(n_gpus):
        cols += ["Power [W]\nmean", "Power [W]\nmax"]

    row = list(range(len(cols)))
    return pd.DataFrame([row], columns=cols)


def test_split_single_gpu_returns_original():
    flat = _make_flat(1)
    result = setup._split_for_logging(flat, n_index_cols=2)
    assert len(result) == 1
    assert list(result[0].columns) == list(flat.columns)


def test_split_no_gpu_returns_original():
    flat = pd.DataFrame([["ModA", "B", 1.0]], columns=["Module", "Backend", "Host\nMem [GB]"])
    result = setup._split_for_logging(flat, n_index_cols=2)
    assert len(result) == 1


def test_split_two_gpus_produces_two_splits():
    flat = _make_flat(2)
    splits = setup._split_for_logging(flat, n_index_cols=2)
    assert len(splits) == 2


def test_split_four_gpus_produces_four_splits():
    flat = _make_flat(4)
    # 4 GPUs → 21 metric cols → ceil(21/6) = 4 splits
    splits = setup._split_for_logging(flat, n_index_cols=2)
    assert len(splits) == 4


def test_split_module_backend_always_present():
    flat = _make_flat(4)
    for split in setup._split_for_logging(flat, n_index_cols=2):
        assert "Module" in split.columns
        assert "Backend" in split.columns


def test_split_host_mem_only_in_first_split():
    flat = _make_flat(4)
    splits = setup._split_for_logging(flat, n_index_cols=2)
    assert "Host\nMem [GB]" in splits[0].columns
    for split in splits[1:]:
        assert "Host\nMem [GB]" not in split.columns


def test_split_each_gpu_appears_in_exactly_one_split():
    n_gpus = 4
    flat = _make_flat(n_gpus)
    splits = setup._split_for_logging(flat, n_index_cols=2)
    for gpu_idx in range(n_gpus):
        mem_col = f"Cuda:{gpu_idx}\nMem [GB]"
        found = sum(1 for s in splits if mem_col in list(s.columns))
        assert found == 1, f"{mem_col} should appear in exactly one split"


def test_split_each_chunk_has_at_most_six_metric_cols():
    flat = _make_flat(5)
    # 5 GPUs → 26 metric cols → ceil(26/6) = 5 splits
    splits = setup._split_for_logging(flat, n_index_cols=2)
    assert len(splits) == 5
    for split in splits:
        assert len(split.columns) - 2 <= 6


def test_dump_metrics_multi_gpu_logs_multiple_tables(caplog, tmp_path, monkeypatch):
    """dump_metrics logs N sub-tables and labels them (1/N) when multiple GPUs are present."""
    import logging

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup, "HARDWARE_METRICS_PATH", None)
    df = pd.DataFrame({
        "module_name": ["ModA"],
        "backend": ["TorchEagerBackend()"],
        "host_memory_used": [1024**3],
        "cuda:0_memory_used": [1024**3],
        "cuda:1_memory_used": [2 * 1024**3],
        "cuda:0_utilization": [50.0],
        "cuda:1_utilization": [60.0],
        "cuda:0_power_usage_milliwatts": [100_000],
        "cuda:1_power_usage_milliwatts": [110_000],
    })
    with caplog.at_level(logging.INFO):
        setup.dump_metrics(df)
    assert "Hardware metrics summary (1/2)" in caplog.text
    assert "Hardware metrics summary (2/2)" in caplog.text


def test_dump_metrics_uses_env_path(caplog, tmp_path, monkeypatch):
    """Test dump_metrics writes to HARDWARE_METRICS_PATH when set."""
    import logging

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup, "HARDWARE_METRICS_PATH", "custom_metrics.csv")
    df = pd.DataFrame({
        "module_name": ["ModA"],
        "backend": ["TorchEagerBackend()"],
        "cpu_memory_used": [1024**3],
    })
    with caplog.at_level(logging.INFO):
        setup.dump_metrics(df)
    assert (tmp_path / "custom_metrics.csv").exists()
    assert "custom_metrics.csv" in caplog.text


def test_dump_metrics_uses_timestamped_name_when_path_not_set(caplog, tmp_path, monkeypatch):
    """Test dump_metrics falls back to timestamped filename when HARDWARE_METRICS_PATH is None."""
    import logging

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup, "HARDWARE_METRICS_PATH", None)
    df = pd.DataFrame({
        "module_name": ["ModA"],
        "backend": ["TorchEagerBackend()"],
        "cpu_memory_used": [1024**3],
    })
    with caplog.at_level(logging.INFO):
        setup.dump_metrics(df)
    csv_files = list(tmp_path.glob("hardware_metrics_*.csv"))
    assert len(csv_files) == 1
