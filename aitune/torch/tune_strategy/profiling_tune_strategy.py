# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared base for profiling-based tune strategies.

A profiling strategy profiles a TorchEager baseline, then builds, validates, and
profiles every user-provided backend, selecting the one whose profiled metric is
best. Subclasses define the metric (throughput, latency, ...) by setting a few
class attributes and implementing a small set of hooks; ``MaxThroughputStrategy``
and ``MinLatencyStrategy`` are the concrete implementations.
"""

import logging
import shutil
import traceback
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from aitune.torch.backend import (
    Backend,
    ONNXRuntimeBackend,
    ONNXRuntimeBackendConfig,
    TensorRTBackend,
    TensorRTBackendConfig,
    TorchEagerBackend,
    TorchInductorAotBackend,
    TorchInductorJitBackend,
    TorchTensorRTAotBackend,
    TorchTensorRTJitBackend,
)
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.task.profiling import ProfilingConfig
from aitune.torch.tune_data.reporting import report_backend_metric, report_graph_baseline_metric
from aitune.torch.tune_strategy.mixin import FindMaxBatchSizeMixin
from aitune.torch.tune_strategy.mixin.performance_validation_mixin import fmt_speedup_msg
from aitune.utils.logging import log


@dataclass
class _TuneCandidate:
    backend: Backend
    value: float
    batch_size: int


@dataclass
class BackendPerfResult:
    """Profiled metric and speedup result for a single backend."""

    backend_description: str
    metric: float
    baseline_metric: float
    speedup: float
    passed: bool


class ProfilingTuneStrategy(FindMaxBatchSizeMixin):
    """Base class for strategies that select a backend by a profiled metric.

    Subclasses set ``_title``, ``_description``, ``_metric_label`` (e.g. "throughput"),
    ``_metric_unit`` (e.g. "samples/s") and ``_value_fmt`` (a ``format`` spec such as ".2f"),
    and implement :meth:`_measure`, :meth:`_is_better`, :meth:`_speedup`,
    :meth:`_report_baseline_value` and :meth:`_report_backend_value`.

    TorchEager is profiled in ``_pre_tune`` as a baseline (not injected into the backends
    list). When performance validation is enabled (default), the strategy falls back to
    TorchEager when no user-provided backend beats it. When disabled, the best
    user-provided backend wins regardless of speed, and the strategy raises if all user
    backends fail.
    """

    _title: str = ""
    _description: str = ""
    _metric_label: str = ""
    _metric_unit: str = ""
    _value_fmt: str = ".2f"

    def __init__(
        self,
        backends: list[Backend] | None = None,
        profiling_config: ProfilingConfig | None = None,
        **kwargs: Any,
    ):
        """Initializes strategy.

        Args:
            backends: List of backends to tune.
            profiling_config: Profiling configuration shared by strategy profiling tasks.
            kwargs: Additional arguments passed to the parent class (e.g. ``sink``).
        """
        super().__init__(profiling_config=profiling_config, **kwargs)
        self._backends = backends or self._default_backends()
        self._performance_validation_enabled: bool = True

        self.perf_validation_results: list[BackendPerfResult] = []
        self._baseline_value: float | None = None
        self._baseline_backend: Backend | None = None
        self._baseline_batch_size: int | None = None

    def enable_performance_validation(self, enable: bool = True) -> "ProfilingTuneStrategy":
        """Enables or disables baseline validation."""
        self._performance_validation_enabled = enable
        return self

    @abstractmethod
    def _measure(
        self,
        backend: Backend,
        name: str,
        graph_spec: GraphSpec,
        data: list[Sample],
        profiling_cfg: ProfilingConfig,
    ) -> tuple[int, float]:
        """Profiles the backend and returns its (batch_size, metric value)."""
        ...

    @abstractmethod
    def _is_better(self, value: float, other: float) -> bool:
        """Returns True when ``value`` is a better metric than ``other``."""
        ...

    @abstractmethod
    def _speedup(self, value: float, baseline_value: float) -> float:
        """Returns the speedup of ``value`` relative to ``baseline_value`` (>1 is faster)."""
        ...

    def _fmt(self, value: float) -> str:
        """Formats a metric value with its unit, e.g. ``12.34 samples/s``."""
        return f"{format(value, self._value_fmt)} {self._metric_unit}"

    def _pre_tune(
        self,
        module: nn.Module,
        name: str,
        graph_spec: GraphSpec,
        data: list[Sample],
        device: torch.device,
        cache_dir: Path,
    ):
        """Calls super()._pre_tune() (finds max batch size) then profiles TorchEager as baseline."""
        super()._pre_tune(module, name, graph_spec, data, device, cache_dir)
        self.perf_validation_results = []
        self._baseline_value = None
        self._baseline_backend = None
        self._baseline_batch_size = None

        if not self._performance_validation_enabled:
            log("⚠️ Performance validation against TorchEager baseline is disabled.", sink=self._sink)
            return

        batching = graph_spec.input_spec.has_batch_axis() and graph_spec.get_max_batch_size() > 1
        max_batch_size = graph_spec.get_max_batch_size()
        profiling_cfg = self._get_profiling_config(batching, max_batch_size)

        baseline_cache_dir = cache_dir / "baseline"
        shutil.rmtree(baseline_cache_dir, ignore_errors=True)
        baseline_cache_dir.mkdir(parents=True)
        try:
            backend = TorchEagerBackend()
            backend = backend.build(module, graph_spec, deepcopy(data), device, baseline_cache_dir)
            batch_size, value = self._measure(backend, name, graph_spec, data, profiling_cfg)
            self._baseline_value = value
            self._baseline_backend = backend
            self._baseline_batch_size = batch_size
            report_graph_baseline_metric(self._metric_label, value)
            log("📊 TorchEager baseline: %s", self._fmt(value), sink=self._sink)
        except Exception:
            error_log = self._log_file(baseline_cache_dir, "error.log")
            error_log.write_text(traceback.format_exc())
            log(
                "⚠️ TorchEager baseline failed (log: %s), performance check skipped",
                error_log,
                sink=self._sink,
            )

    def _tune(
        self,
        module: nn.Module,
        name: str,
        graph_spec: GraphSpec,
        data: list[Sample],
        device: torch.device,
        cache_dir: Path,
    ) -> Backend:
        """Tunes given torch module with provided graph_spec and data."""
        log(
            "⏳ Executing strategy `%s` on module `%s` (graph: %s)",
            self.__class__.__name__,
            name,
            graph_spec.name,
            sink=self._sink,
        )
        batching = graph_spec.input_spec.has_batch_axis() and graph_spec.get_max_batch_size() > 1
        max_batch_size = graph_spec.get_max_batch_size()

        best = self._run_backends(module, name, graph_spec, data, device, cache_dir, batching, max_batch_size)
        winner = self._resolve_winner(best)
        winner.backend.activate()
        log("🎯 Strategy %s execution finished:", self.__class__.__name__, sink=self._sink)
        log(
            "✅ Selected %s for module %s and graph spec %s.",
            winner.backend.describe(),
            name,
            graph_spec,
            sink=self._sink,
        )
        log(
            "   Batch size: %s, %s: %s", winner.batch_size, self._metric_label, self._fmt(winner.value), sink=self._sink
        )
        return winner.backend

    def _run_backends(
        self,
        module: nn.Module,
        name: str,
        graph_spec: GraphSpec,
        data: list[Sample],
        device: torch.device,
        cache_dir: Path,
        batching: bool,
        max_batch_size: int,
    ) -> _TuneCandidate | None:
        """Builds, validates, and profiles each backend; returns the best candidate."""
        best: _TuneCandidate | None = None

        for backend in self._backends:
            log_file = self._log_file(cache_dir / backend.key(), "build.log")
            built = self._build_and_validate_backend(backend, module, name, graph_spec, data, device, cache_dir)
            if built is None:
                continue
            try:
                batch_size, value = self._measure(
                    built, name, graph_spec, data, self._get_profiling_config(batching, max_batch_size)
                )
                self.backend_results[-1].update({self._metric_label: value, "max_batch_size": batch_size})
                report_backend_metric(self._metric_label, built.describe(), value)
                log(
                    "✅ backend profiled - %s: %s, batch size: %s",
                    self._metric_label,
                    self._fmt(value),
                    batch_size,
                    depth=2,
                    sink=self._sink,
                )
                self._record_perf_result(built, value)
                if best is None or self._is_better(value, best.value):
                    if best is not None and best.backend.is_active:
                        best.backend.deactivate()
                    log(
                        "🎯 new best %s for %s is %s, batch size: %s",
                        self._metric_label,
                        built.describe(),
                        self._fmt(value),
                        batch_size,
                        depth=2,
                        sink=self._sink,
                    )
                    best = _TuneCandidate(backend=built, value=value, batch_size=batch_size)
                else:
                    if built.is_active:
                        built.deactivate()
            except Exception:
                if built.is_active:
                    built.deactivate()
                log("❌ backend failed (log file: %s)", log_file, depth=2, sink=self._sink)

        return best

    def _record_perf_result(self, backend: Backend, value: float) -> None:
        """Appends a BackendPerfResult for the given backend if a baseline is available."""
        if self._baseline_value is None or self._baseline_value <= 0 or value <= 0:
            return
        speedup = self._speedup(value, self._baseline_value)
        self.perf_validation_results.append(
            BackendPerfResult(
                backend_description=backend.describe(),
                metric=value,
                baseline_metric=self._baseline_value,
                speedup=speedup,
                passed=speedup >= 1.0,
            )
        )

    def _resolve_winner(self, best: _TuneCandidate | None) -> _TuneCandidate:
        """Returns the winning candidate, falling back to the TorchEager baseline when appropriate."""
        use_baseline = (
            self._performance_validation_enabled
            and self._baseline_backend is not None
            and self._baseline_value is not None
            and (best is None or self._is_better(self._baseline_value, best.value))
        )
        if use_baseline:
            if best is not None and best.backend.is_active:
                best.backend.deactivate()
            reason = (
                "no user backend succeeded"
                if best is None
                else f"best user backend ({self._fmt(best.value)}) slower than baseline"
            )
            log(
                "ℹ️ Falling back to TorchEager baseline (%s): %s",
                self._fmt(self._baseline_value),
                reason,
                sink=self._sink,
            )
            return _TuneCandidate(
                backend=self._baseline_backend,
                value=self._baseline_value,
                batch_size=self._baseline_batch_size or 1,
            )
        if best is None:
            log(
                "ℹ️ No correct backend found. Backends considered: %s",
                ", ".join([b.describe() for b in self._backends]),
                sink=self._sink,
            )
            raise RuntimeError("No correct backend found")
        return best

    def _post_tune(self, backend: Backend | None, name: str, graph_spec: GraphSpec, data: list[Sample]):
        """Emits a speedup line after tuning completes."""
        super()._post_tune(backend, name, graph_spec, data)
        if backend is None:
            return
        result = next(
            (r for r in self.perf_validation_results if r.backend_description == backend.describe()),
            None,
        )
        if result is None:
            if backend is self._baseline_backend:
                self._log_baseline_selected(backend)
            return
        detail = f"{format(result.baseline_metric, self._value_fmt)} → {self._fmt(result.metric)}"
        msg_short = fmt_speedup_msg(result.speedup, detail, name, backend.describe())
        if self._logger.isEnabledFor(logging.INFO):
            log(msg_short, sink=self._sink)
        else:
            msg_full = f"[AITune] {msg_short}"
            log(msg_full, sink=self._logger.warning)

    def _log_baseline_selected(self, backend: Backend) -> None:
        """Emit an explicit message when the TorchEager baseline is the selected backend."""
        value = f" ({self._fmt(self._baseline_value)})" if self._baseline_value is not None else ""
        msg = f"ℹ️ Baseline was selected: {backend.describe()}{value}"
        if not self._logger.isEnabledFor(logging.INFO):
            return
        log(msg, sink=self._sink)

    def _get_profiling_config(self, batching: bool, max_batch_size: int) -> ProfilingConfig:
        """Gets profiling configuration."""
        batch_sizes = sorted({
            batch_size for batch_size in self.profiling_config.batch_sizes if batch_size <= max_batch_size
        })
        return replace(
            self.profiling_config,
            batch_sizes=batch_sizes or [max_batch_size],
            batching=batching,
        )

    def _default_backends(self) -> list[Backend]:
        """Returns default backends."""
        return [
            TensorRTBackend(),
            TensorRTBackend(config=TensorRTBackendConfig(use_dynamo=False)),
            TorchInductorJitBackend(),
            TorchInductorAotBackend(),
            TorchTensorRTJitBackend(),
            TorchTensorRTAotBackend(),
            ONNXRuntimeBackend(),
            ONNXRuntimeBackend(config=ONNXRuntimeBackendConfig(use_dynamo=False)),
        ]

    def _describe_parts(self) -> list[str]:
        """Returns the parts of the description."""
        return [
            f"name: {self._title}",
            f"description: {self._description}",
            "backends:",
            *[f"  {backend.describe()}" for backend in self._backends],
        ]

    def to_json_dict(self) -> dict[str, Any]:
        """Returns config dict for the strategy."""
        return {
            "backends": [b.describe() for b in self._backends],
            "profiling_config": self._profiling_config_to_json_dict(),
        }
