# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardware metrics collector."""

import atexit
import logging
import multiprocessing as mp
from datetime import datetime

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

_DEFAULT_COLLECTOR: NoOpHardwareMetricsCollector | None = None
_DEFAULT_RECEIVER: ParentReceiver | None = None


def get_default_collector() -> NoOpHardwareMetricsCollector:
    """Get the default hardware metrics collector."""
    global _DEFAULT_COLLECTOR, _DEFAULT_RECEIVER

    if _DEFAULT_COLLECTOR is None:
        if HARDWARE_METRICS_ENABLED:
            queue = mp.Queue()
            metric_providers = [HostMetricProvider(), TorchMetricProvider()]
            try:
                metric_providers.append(NvmlMetricProvider())
            except NVMLError:
                logging.warning("NVML not available, skipping GPU metrics")
            _DEFAULT_COLLECTOR = HardwareMetricsCollector(metric_providers, queue, interval=0.1)
            _DEFAULT_RECEIVER = ParentReceiver(queue=queue)
            _DEFAULT_RECEIVER.start()

            def dump_metrics_and_stop_receiver():
                """Dumps metrics and stops the receiver process."""
                if _DEFAULT_RECEIVER is not None:
                    metrics = _DEFAULT_RECEIVER.get_metrics()
                    dump_metrics(metrics)
                    _DEFAULT_RECEIVER.stop()

            atexit.register(dump_metrics_and_stop_receiver)
        else:
            _DEFAULT_COLLECTOR = NoOpHardwareMetricsCollector()

    return _DEFAULT_COLLECTOR


def get_hardware_metrics() -> pd.DataFrame | None:
    """Get metrics from the receiver."""
    if _DEFAULT_RECEIVER is not None:
        return _DEFAULT_RECEIVER.get_metrics()
    return None


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
    metrics.to_csv(metrics_file, index=False)
    logging.info("Dumped hardware metrics to CSV file: %s", metrics_file)
