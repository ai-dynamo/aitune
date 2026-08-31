# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared base for profiling-based tune strategies.

A profiling strategy profiles a TorchEager baseline, then builds, validates, and
profiles every user-provided backend, selecting the one whose profiled metric is
best. Subclasses define the metric (throughput, latency, ...) by setting a few
class attributes and implementing a small set of hooks; ``MaxThroughputStrategy``,
``MinLatencyStrategy``, and ``LatencyBudgetStrategy`` are the concrete implementations.
"""

import logging
import shutil
import traceback
from abc import ABC, abstractmethod
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
from aitune.torch.distributed import coordinator
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import SampleStore
from aitune.torch.task.profiling import ProfilingConfig
from aitune.torch.tune_data.reporting import report_backend_metric, report_graph_baseline_metric
from aitune.torch.tune_strategy.mixin import FindMaxBatchSizeMixin
from aitune.torch.tune_strategy.mixin.performance_validation_mixin import fmt_speedup_msg
from aitune.utils.logging import log


@dataclass
class _TuneCandidate:
    backend: Backend
    result: "BackendProfilingResult"


@dataclass(kw_only=True)
class BackendProfilingResult(ABC):
    """Profiled backend result returned by profiling strategies."""

    selected_batch_size: int

    @property
    @abstractmethod
    def metric(self) -> float:
        """Returns the scalar metric used to compare candidates."""
        ...

    def to_json_dict(self, metric_label: str) -> dict[str, int | float]:
        """Returns fields stored in strategy_results for this backend."""
        return {
            metric_label: self.metric,
            "selected_batch_size": self.selected_batch_size,
        }


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
    and implement :meth:`_measure`, :meth:`_is_better`, and :meth:`_speedup`.

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
        self._backends = backends if backends is not None else self._default_backends()
        self._performance_validation_enabled: bool = True

        self.perf_validation_results: list[BackendPerfResult] = []
        self._baseline_backend: Backend | None = None
        self._baseline_result: BackendProfilingResult | None = None

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
        samples: SampleStore,
        profiling_cfg: ProfilingConfig,
    ) -> BackendProfilingResult:
        """Profiles the backend and returns its result."""
        ...

    @abstractmethod
    def _is_better(self, result: BackendProfilingResult, other: BackendProfilingResult) -> bool:
        """Returns True when ``result`` is better than ``other``."""
        ...

    @abstractmethod
    def _speedup(self, result: BackendProfilingResult, baseline_result: BackendProfilingResult) -> float:
        """Returns the speedup of ``result`` relative to ``baseline_result`` (>1 is faster)."""
        ...

    def _fmt(self, value: float) -> str:
        """Formats a metric value with its unit, e.g. ``12.34 samples/s``."""
        return f"{format(value, self._value_fmt)} {self._metric_unit}"

    def _aggregate_result(
        self,
        gathered_results: list[BackendProfilingResult],
    ) -> BackendProfilingResult:
        """Aggregate profiling results into a deterministic distributed result."""
        result = gathered_results[0]
        batch_sizes = [candidate.selected_batch_size for candidate in gathered_results]
        if any(batch_size != batch_sizes[0] for batch_size in batch_sizes[1:]):
            details = ", ".join(f"rank {rank}: {batch_size!r}" for rank, batch_size in enumerate(batch_sizes))
            raise RuntimeError(f"Distributed selected profiling batch size differs across ranks: {details}")

        updates = {}
        if hasattr(result, "throughput"):
            updates["throughput"] = min(candidate.throughput for candidate in gathered_results)
        if hasattr(result, "latency"):
            updates["latency"] = max(candidate.latency for candidate in gathered_results)
        if updates:
            return replace(result, **updates)

        worst = gathered_results[0]
        for candidate in gathered_results[1:]:
            if self._is_better(worst, candidate):
                worst = candidate
        return worst

    def _measure_distributed(
        self,
        backend: Backend,
        name: str,
        graph_spec: GraphSpec,
        samples: SampleStore,
        profiling_cfg: ProfilingConfig,
    ) -> BackendProfilingResult | None:
        """Measure locally and return an aggregated result only if every rank succeeds."""
        local_error: Exception | None = None
        result: BackendProfilingResult | None = None
        try:
            result = self._measure(backend, name, graph_spec, samples, profiling_cfg)
        except Exception as e:
            local_error = e
        outcome, gathered_results = coordinator.collect_results(result, local_error)
        if not outcome.succeeded:
            return None
        assert result is not None
        return self._aggregate_result(gathered_results)

    def _pre_tune(
        self,
        module: nn.Module,
        name: str,
        graph_spec: GraphSpec,
        samples: SampleStore,
        device: torch.device,
        cache_dir: Path,
    ):
        """Calls super()._pre_tune() (finds max batch size) then profiles TorchEager as baseline."""
        super()._pre_tune(module, name, graph_spec, samples, device, cache_dir)
        self.perf_validation_results = []
        self._baseline_backend = None
        self._baseline_result = None

        if not self._performance_validation_enabled:
            log("⚠️ Performance validation against TorchEager baseline is disabled.", sink=self._sink)
            return

        batching = graph_spec.input_spec.has_batch_axis() and graph_spec.get_max_batch_size() > 1
        max_batch_size = graph_spec.get_max_batch_size()
        profiling_cfg = self._get_profiling_config(batching, max_batch_size)

        baseline_cache_dir = cache_dir / "baseline"
        shutil.rmtree(baseline_cache_dir, ignore_errors=True)
        baseline_cache_dir.mkdir(parents=True)
        local_error: Exception | None = None
        backend = TorchEagerBackend()
        result: BackendProfilingResult | None = None
        try:
            with coordinator.raise_if_any_rank_fails("Building TorchEager baseline"):
                backend = backend.build(module, graph_spec, samples, device, baseline_cache_dir)
            result = self._measure(backend, name, graph_spec, samples, profiling_cfg)
        except Exception as e:
            local_error = e
            error_log = self._log_file(baseline_cache_dir, "error.log")
            error_log.write_text("".join(traceback.format_exception(e)))

        outcome, gathered_results = coordinator.collect_results(result, local_error)
        if not outcome.succeeded:
            if backend.is_active:
                backend.deactivate()
            log(
                "⚠️ TorchEager baseline failed (log: %s), performance check skipped",
                self._log_file(baseline_cache_dir, "error.log"),
                sink=self._sink,
            )
            return

        assert result is not None
        result = self._aggregate_result(gathered_results)
        self._baseline_backend = backend
        self._baseline_result = result
        report_graph_baseline_metric(self._metric_label, result.metric)
        log("📊 TorchEager baseline: %s", self._fmt(result.metric), sink=self._sink)

    def _tune(
        self,
        module: nn.Module,
        name: str,
        graph_spec: GraphSpec,
        samples: SampleStore,
        device: torch.device,
        cache_dir: Path,
    ) -> Backend:
        """Tune a torch module with the provided graph specification and samples."""
        log(
            "⏳ Executing strategy `%s` on module `%s` (graph: %s)",
            self.__class__.__name__,
            name,
            graph_spec.name,
            sink=self._sink,
        )
        batching = graph_spec.input_spec.has_batch_axis() and graph_spec.get_max_batch_size() > 1
        max_batch_size = graph_spec.get_max_batch_size()

        best = self._run_backends(module, name, graph_spec, samples, device, cache_dir, batching, max_batch_size)
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
            "   Batch size: %s, %s: %s",
            winner.result.selected_batch_size,
            self._metric_label,
            self._fmt(winner.result.metric),
            sink=self._sink,
        )
        return winner.backend

    def _run_backends(
        self,
        module: nn.Module,
        name: str,
        graph_spec: GraphSpec,
        samples: SampleStore,
        device: torch.device,
        cache_dir: Path,
        batching: bool,
        max_batch_size: int,
    ) -> _TuneCandidate | None:
        """Builds, validates, and profiles each backend; returns the best candidate."""
        best: _TuneCandidate | None = None

        for backend in self._backends:
            log_file = self._log_file(cache_dir / backend.key(), "build.log")
            built = self._build_and_validate_backend(backend, module, name, graph_spec, samples, device, cache_dir)
            if built is None:
                continue
            result = self._measure_distributed(
                built,
                name,
                graph_spec,
                samples,
                self._get_profiling_config(batching, max_batch_size),
            )
            if result is None:
                if built.is_active:
                    built.deactivate()
                log("❌ backend failed (log file: %s)", log_file, depth=2, sink=self._sink)
                continue

            try:
                self.backend_results[-1].update(result.to_json_dict(self._metric_label))
                report_backend_metric(self._metric_label, built.describe(), result.metric)
                batch_size = result.selected_batch_size
                log(
                    "✅ backend profiled - %s: %s, batch size: %s",
                    self._metric_label,
                    self._fmt(result.metric),
                    batch_size,
                    depth=2,
                    sink=self._sink,
                )
                self._record_perf_result(built, result)
                if best is None or self._is_better(result, best.result):
                    if best is not None and best.backend.is_active:
                        best.backend.deactivate()
                    log(
                        "🎯 new best %s for %s is %s, batch size: %s",
                        self._metric_label,
                        built.describe(),
                        self._fmt(result.metric),
                        batch_size,
                        depth=2,
                        sink=self._sink,
                    )
                    best = _TuneCandidate(backend=built, result=result)
                else:
                    if built.is_active:
                        built.deactivate()
            except Exception:
                if built.is_active:
                    built.deactivate()
                log("❌ backend failed (log file: %s)", log_file, depth=2, sink=self._sink)

        return best

    def _record_perf_result(self, backend: Backend, result: BackendProfilingResult) -> None:
        """Appends a BackendPerfResult for the given backend if a baseline is available."""
        metric = result.metric
        if self._baseline_result is None or self._baseline_result.metric <= 0 or metric <= 0:
            return
        speedup = self._speedup(result, self._baseline_result)
        self.perf_validation_results.append(
            BackendPerfResult(
                backend_description=backend.describe(),
                metric=metric,
                baseline_metric=self._baseline_result.metric,
                speedup=speedup,
                passed=speedup >= 1.0,
            )
        )

    def _resolve_winner(self, best: _TuneCandidate | None) -> _TuneCandidate:
        """Returns the winning candidate, falling back to the TorchEager baseline when appropriate."""
        use_baseline = (
            self._performance_validation_enabled
            and self._baseline_backend is not None
            and self._baseline_result is not None
            and (best is None or not self._is_better(best.result, self._baseline_result))
        )
        if use_baseline:
            if best is not None and best.backend.is_active:
                best.backend.deactivate()
            reason = (
                "no user backend succeeded"
                if best is None
                else f"best user backend ({self._fmt(best.result.metric)}) did not beat baseline"
            )
            log(
                "ℹ️ Falling back to TorchEager baseline (%s): %s",
                self._fmt(self._baseline_result.metric),
                reason,
                sink=self._sink,
            )
            return _TuneCandidate(
                backend=self._baseline_backend,
                result=self._baseline_result,
            )
        if best is None:
            log(
                "ℹ️ No correct backend found. Backends considered: %s",
                ", ".join([b.describe() for b in self._backends]),
                sink=self._sink,
            )
            raise RuntimeError("No correct backend found")
        return best

    def _post_tune(self, backend: Backend | None, name: str, graph_spec: GraphSpec, samples: SampleStore):
        """Emits a speedup line after tuning completes."""
        super()._post_tune(backend, name, graph_spec, samples)
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
        value = f" ({self._fmt(self._baseline_result.metric)})" if self._baseline_result is not None else ""
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
