# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardware metrics collector."""

import threading
from collections import deque
from multiprocessing.queues import Queue as mpQueue
from queue import Queue as plainQueue
from time import perf_counter
from typing import Any

from aitune.global_context import BACKEND_CONTEXT_KEY, MODULE_CONTEXT_KEY, global_context
from aitune.utils.monitoring.hardware_metrics.metrics import (
    AbstractHardwareMetricProvider,
)


class NoOpHardwareMetricsCollector:
    """No-op hardware metrics collector used when metric collection is disabled.

    Both methods intentionally do nothing. This is the null-object counterpart to
    HardwareMetricsCollector and can be used wherever a collector is required but
    collecting metrics is not desired.
    """

    def start_scope(self, name: str) -> None:
        """Starts a new scope."""
        pass

    def end_scope(self) -> None:
        """Ends the current scope."""
        pass


class HardwareMetricsCollector(NoOpHardwareMetricsCollector):
    """Hardware metrics collector that collects metrics from the GPU and CPU.

    This class collects metrics from the GPU and CPU at a given interval using a timer.
    The results are sent to a queue to be processed by the hardware metrics receiver.
    The metrics are collected only if there is any scope active.
    """

    def __init__(
        self,
        metric_providers: list[AbstractHardwareMetricProvider],
        queue: plainQueue | mpQueue,
        interval: float = 0.1,
    ):
        """Initializes the hardware metrics collector."""
        self.metric_providers = metric_providers
        self.interval = interval
        self.queue = queue

        self.lock = threading.RLock()
        self.scopes = deque[str]()
        self.timer = None
        self.start_time = perf_counter()
        self.previous_polled_scope = None

    def start_scope(self, name: str) -> None:
        """Starts a new scope."""
        with self.lock:
            self.scopes.append(name)
            if self.timer is None:
                # start a polling
                self._poll_stats()
            else:
                if self.previous_polled_scope != name:
                    # timer is polling at its pace, if it has not yet polled the current scope, queue metrics immediately
                    # so that short lived scopes are not missed
                    self._queue_metrics()

    def end_scope(self) -> None:
        """Ends the current scope."""
        with self.lock:
            if not self.scopes:
                raise RuntimeError("end_scope called without a matching start_scope")
            self.scopes.pop()
            if not self.scopes:
                self._cancel_polling()

    def _poll_stats(self) -> None:
        """Polls the metrics."""
        curr_time = perf_counter()
        with self.lock:
            self._queue_metrics()
            processing_time = perf_counter() - curr_time
            interval = self.interval - processing_time if processing_time < self.interval else 0.0
            self.timer = threading.Timer(interval, self._poll_stats)
            self.timer.start()

    def _queue_metrics(self):
        """Queues the metrics."""
        with self.lock:
            if not self.scopes:
                return  # Scope ended between timer dispatch and execution
            metrics = self._collect_metrics()
            self._update_metrics_with_context_info(metrics, perf_counter())
            self.queue.put(metrics)
            self.previous_polled_scope = self.scopes[-1]

    def _cancel_polling(self):
        """Cancels the polling."""
        with self.lock:
            if self.timer:
                self.timer.cancel()
                self.timer = None

    def _collect_metrics(self) -> dict[str, Any]:
        """Collects metrics from the GPU and CPU."""
        metrics = {}
        for provider in self.metric_providers:
            metrics.update(provider.get_metrics())
        return metrics

    def _update_metrics_with_context_info(self, metrics: dict[str, Any], curr_time: float) -> None:
        """Updates collected metrics with context information.

        Args:
            metrics: The metrics dictionary to update.
            curr_time: The current timestamp.
        """
        metrics["scope"] = self.scopes[-1]
        metrics["module_name"] = global_context.get(MODULE_CONTEXT_KEY)
        metrics["backend"] = global_context.get(BACKEND_CONTEXT_KEY)
        metrics["timestamp"] = curr_time - self.start_time
