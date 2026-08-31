# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardware metrics collector."""

import atexit
import logging
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import pandas as pd
from pynvml import NVMLError
from tabulate import tabulate

from aitune.utils.env_vars import HARDWARE_METRICS_ENABLED, HARDWARE_METRICS_PATH
from aitune.utils.monitoring.hardware_metrics.collector import (
    HardwareMetricsCollector,
    NoOpHardwareMetricsCollector,
)
from aitune.utils.monitoring.hardware_metrics.metrics import (
    HostMetricProvider,
    NvmlMetricProvider,
    TorchMetricProvider,
)
from aitune.utils.monitoring.hardware_metrics.parent_receiver import ParentReceiver
from aitune.utils.monitoring.parse_metrics import format_backend_label_for_display, get_metrics_summary


class HardwareMetricsSession:
    """Manages the lifecycle of hardware metrics collection.

    Encapsulates the collector, receiver subprocess, runtime enable/disable state,
    and atexit registration.
    """

    def __init__(self):
        """Initializes the session in a disabled state."""
        self._collector: HardwareMetricsCollector | None = None
        self._receiver: ParentReceiver | None = None
        self._runtime_enabled: bool | None = None  # None defers to HARDWARE_METRICS_ENABLED
        self._noop_collector = NoOpHardwareMetricsCollector()
        self._atexit_handler = None

    def _should_collect(self) -> bool:
        return self._runtime_enabled if self._runtime_enabled is not None else HARDWARE_METRICS_ENABLED

    def _ensure_initialized(self) -> None:
        if self._collector is not None:
            return

        queue = mp.Queue()
        metric_providers = [HostMetricProvider(), TorchMetricProvider()]
        try:
            metric_providers.append(NvmlMetricProvider())
        except NVMLError:
            logging.warning("NVML not available, skipping GPU metrics")
        self._collector = HardwareMetricsCollector(metric_providers, queue, interval=0.1)
        self._receiver = ParentReceiver(queue=queue)
        self._receiver.start()

        def _dump_and_stop():
            if self._receiver is not None:
                metrics = self._receiver.get_metrics()
                dump_metrics(metrics)
                self._receiver.stop()

        self._atexit_handler = _dump_and_stop
        atexit.register(_dump_and_stop)

    # ── public interface ──────────────────────────────────────────────────────

    def get_collector(self) -> NoOpHardwareMetricsCollector:
        """Return the active collector, or a no-op when collection is disabled."""
        if not self._should_collect():
            return self._noop_collector
        self._ensure_initialized()
        return self._collector  # type: ignore[return-value]

    def enable(self) -> None:
        """Enable collection at runtime, overriding the env var."""
        self._runtime_enabled = True

    def disable(self) -> None:
        """Disable collection at runtime, gracefully stopping any active session."""
        self._runtime_enabled = False
        if self._receiver is not None:
            if self._atexit_handler is not None:
                atexit.unregister(self._atexit_handler)
                self._atexit_handler = None
            metrics = self._receiver.get_metrics()
            dump_metrics(metrics)
            self._receiver.stop()
            self._receiver = None
            self._collector = None

    def snapshot(self, path: Path, reset_metrics: bool = True) -> None:
        """Dump currently collected metrics to a CSV file.

        Args:
            path: Path to write the CSV file.
            reset_metrics: If True, clears the accumulated metrics after writing.
        """
        if self._receiver is None:
            logging.warning("Hardware metrics collection is not active; snapshot skipped.")
            return
        self._receiver.snapshot(path, reset_metrics)
        logging.info("Hardware metrics snapshot written to %s", path)

    def get_metrics(self) -> pd.DataFrame | None:
        """Return accumulated metrics, or None if collection was never started."""
        if self._receiver is not None:
            return self._receiver.get_metrics()
        return None


_DEFAULT_SESSION: HardwareMetricsSession = HardwareMetricsSession()


def get_default_collector() -> NoOpHardwareMetricsCollector:
    """Get the default hardware metrics collector."""
    return _DEFAULT_SESSION.get_collector()


def enable_hardware_metrics() -> None:
    """Enable hardware metrics collection at runtime.

    Overrides the AITUNE_HARDWARE_METRICS environment variable.
    The collector and receiver subprocess are started lazily on the first use.
    """
    _DEFAULT_SESSION.enable()


def disable_hardware_metrics() -> None:
    """Disable hardware metrics collection at runtime.

    Overrides the AITUNE_HARDWARE_METRICS environment variable.
    If collection was active, gracefully stops the receiver (dumping any
    accumulated metrics) and resets state so that a future enable_hardware_metrics()
    call starts a fresh collection session.
    """
    _DEFAULT_SESSION.disable()


def snapshot(path: Path, reset_metrics: bool = True) -> None:
    """Dump currently collected hardware metrics to a CSV file.

    Args:
        path: Path to write the CSV file.
        reset_metrics: If True, clears the accumulated metrics after writing.
    """
    _DEFAULT_SESSION.snapshot(path, reset_metrics)


def get_hardware_metrics() -> pd.DataFrame | None:
    """Get metrics from the receiver."""
    return _DEFAULT_SESSION.get_metrics()


def _split_for_logging(flat: pd.DataFrame, n_index_cols: int) -> list[pd.DataFrame]:
    """Split a wide metrics table into narrower sub-tables for logging.

    Each split contains the index columns (Module, Backend) plus up to 6
    metric columns. Returns the original frame unchanged when there are at
    most 6 metric columns.
    """
    n_metric_cols = len(flat.columns) - n_index_cols
    index_indices = list(range(n_index_cols))

    chunk_size = 6
    if n_metric_cols <= chunk_size:
        return [flat]

    splits = []
    for i in range(0, n_metric_cols, chunk_size):
        col_indices = index_indices + list(range(n_index_cols + i, n_index_cols + min(i + chunk_size, n_metric_cols)))
        splits.append(flat.iloc[:, col_indices])
    return splits


def dump_metrics(metrics: pd.DataFrame):
    """Logs metrics summary and dumps the metrics to a CSV file.

    Args:
        metrics: The metrics to dump.
    """
    df_summary = get_metrics_summary(metrics)
    if df_summary is None:
        return
    flat = df_summary.reset_index()
    flat["Backend"] = flat["Backend"].apply(format_backend_label_for_display)
    n_index_cols = df_summary.index.nlevels

    splits = _split_for_logging(flat, n_index_cols)
    n_splits = len(splits)
    for split_idx, split in enumerate(splits):
        col_align = ("left",) * n_index_cols + ("center",) * (len(split.columns) - n_index_cols)
        label = f" ({split_idx + 1}/{n_splits})" if n_splits > 1 else ""
        logging.info(
            "Hardware metrics summary%s:\n%s",
            label,
            tabulate(split, headers="keys", tablefmt="fancy_grid", colalign=col_align, showindex=False),
        )
    if HARDWARE_METRICS_PATH is not None:
        metrics_file = HARDWARE_METRICS_PATH
    else:
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = f"hardware_metrics_{date_time}.csv"
    from aitune.torch.distributed import distributed_output_path

    metrics_file = distributed_output_path(metrics_file)
    metrics.to_csv(metrics_file, index=False)
    logging.info("Dumped hardware metrics to CSV file: %s", metrics_file)
