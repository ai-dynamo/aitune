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
import math
import shutil
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import geometric_mean
from typing import Any, Literal

import torch
import torch.nn as nn

from aitune.exceptions import AITuneUserInputError
from aitune.torch.backend import Backend
from aitune.torch.distributed import coordinator
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.onnx_module import OnnxModule
from aitune.torch.module.sample_store import SampleStore
from aitune.torch.task.profiling import ProfilingConfig
from aitune.torch.tune_data.reporting import report_backend_metric, report_graph_baseline_metric
from aitune.torch.tune_strategy.formatting import fmt_speedup_msg
from aitune.torch.tune_strategy.multi_backend_strategy import MultiBackendStrategy
from aitune.torch.tune_strategy.performance_validation import PerformanceValidationMode
from aitune.torch.utils.module import get_default_backend_for_module
from aitune.utils.logging import log


@dataclass
class _TuneCandidate:
    backend: Backend
    result: "BackendProfilingResult"
    boundary_results: dict[str, "BackendProfilingResult"] = field(default_factory=dict)
    speedup: float | None = None


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
    boundary_speedups: dict[str, float] = field(default_factory=dict)


class ProfilingTuneStrategy(MultiBackendStrategy):
    """Base class for strategies that select a backend by a profiled metric.

    Subclasses set ``_title``, ``_description``, ``_metric_label`` (e.g. "throughput"),
    ``_metric_unit`` (e.g. "samples/s") and ``_value_fmt`` (a ``format`` spec such as ".2f"),
    and implement :meth:`_measure`, :meth:`_is_better`, and :meth:`_speedup`.

    TorchEager is profiled in ``_pre_tune`` as a baseline (not injected into the backends
    list) unless performance validation is disabled. In enabled mode (default), the
    strategy rejects candidates below ``min_boundary_speedup`` at either boundary,
    then selects the highest geometric-mean speedup, with eager scoring 1.0.
    Diagnostic mode ranks all measured candidates without the eager gate. Disabled
    mode skips eager profiling and ranks candidates by their boundary metrics.
    """

    _title: str = ""
    _description: str = ""
    _metric_label: str = ""
    _metric_unit: str = ""
    _value_fmt: str = ".2f"

    def __init__(
        self,
        backends: list[Backend],
        profiling_config: ProfilingConfig | None = None,
        min_boundary_speedup: float = 0.95,
        **kwargs: Any,
    ):
        """Initializes strategy.

        Args:
            backends: List of backends to tune.
            profiling_config: Profiling configuration shared by strategy profiling tasks.
            min_boundary_speedup: Minimum allowed speedup against eager at each shape boundary.
            kwargs: Additional arguments passed to the parent class (e.g. ``sink``).
        """
        super().__init__(backends=backends, profiling_config=profiling_config, **kwargs)
        self._performance_validation_mode = PerformanceValidationMode.ENABLED
        if not math.isfinite(min_boundary_speedup) or min_boundary_speedup <= 0:
            raise AITuneUserInputError("min_boundary_speedup must be finite and positive")
        self.min_boundary_speedup = min_boundary_speedup

        self.perf_validation_results: list[BackendPerfResult] = []
        self._baseline_backend: Backend | None = None
        self._baseline_result: BackendProfilingResult | None = None
        self._baseline_boundary_results: dict[str, BackendProfilingResult] = {}

    def enable_performance_validation(
        self, mode: PerformanceValidationMode | str | bool = PerformanceValidationMode.ENABLED
    ) -> "ProfilingTuneStrategy":
        """Set how eager-baseline performance data affects backend selection.

        Args:
            mode: ``ENABLED`` applies the eager comparison as a selection gate, ``DIAGNOSTIC`` records the
                comparison without affecting selection, and ``DISABLED`` skips eager-baseline profiling.
                For compatibility, ``True`` maps to ``ENABLED`` and ``False`` maps to ``DISABLED``.
        """
        if isinstance(mode, bool):
            mode = PerformanceValidationMode.ENABLED if mode else PerformanceValidationMode.DISABLED
        self._performance_validation_mode = PerformanceValidationMode(mode)
        return self

    @property
    def performance_validation_mode(self) -> PerformanceValidationMode:
        """Return the configured performance validation mode."""
        return self._performance_validation_mode

    def to_json_dict(self) -> dict[str, Any]:
        """Returns config dict for the strategy."""
        return {
            "backends": [b.describe() for b in self._backends],
            "performance_validation_mode": self._performance_validation_mode.value,
            "profiling_config": self._profiling_config_to_json_dict(),
            "min_boundary_speedup": self.min_boundary_speedup,
        }

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

    def _fmt_result(self, result: BackendProfilingResult) -> str:
        """Include mean latency when a throughput result also records it."""
        formatted = self._fmt(result.metric)
        metrics = result.to_json_dict(self._metric_label)
        if self._metric_label == "throughput" and "latency" in metrics:
            formatted += f", mean latency: {metrics['latency']:.3f} ms"
        return formatted

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
        metrics = [candidate.to_json_dict(self._metric_label) for candidate in gathered_results]
        if "throughput" in metrics[0]:
            updates["throughput"] = min(candidate["throughput"] for candidate in metrics)
        if "latency" in metrics[0]:
            updates["latency"] = max(candidate["latency"] for candidate in metrics)
        if updates:
            return replace(result, selected_batch_size=result.selected_batch_size, **updates)

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
        if result is None:
            raise RuntimeError("Profiling succeeded on all ranks but produced no result")
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
        module_type = "Onnx" if isinstance(module, OnnxModule) else "TorchEager"
        self.perf_validation_results = []
        self._baseline_backend = None
        self._baseline_result = None
        self._baseline_boundary_results = {}

        if self._performance_validation_mode is PerformanceValidationMode.DISABLED:
            log("⚠️ Performance validation against TorchEager baseline is disabled.", sink=self._sink)
            return

        if self._performance_validation_mode is PerformanceValidationMode.DIAGNOSTIC:
            log(
                "ℹ️ Performance validation is diagnostic-only; eager comparison will not affect backend selection.",
                sink=self._sink,
            )

        batching = graph_spec.input_spec.has_batch_axis() and graph_spec.get_max_batch_size() > 1
        max_batch_size = graph_spec.get_max_batch_size()
        profiling_cfg = self._get_profiling_config(batching, max_batch_size)

        baseline_cache_dir = cache_dir / "baseline"
        shutil.rmtree(baseline_cache_dir, ignore_errors=True)
        baseline_cache_dir.mkdir(parents=True)
        local_error: Exception | None = None
        backend = get_default_backend_for_module(module)
        result: BackendProfilingResult | None = None
        try:
            with coordinator.raise_if_any_rank_fails("Building TorchEager baseline"):
                backend = backend.build(module, graph_spec, samples, device, baseline_cache_dir)
            self._baseline_boundary_results = self._measure_boundaries(
                backend, name, graph_spec, samples, profiling_cfg
            )
            result = self._baseline_boundary_results.get("max_shape", self._baseline_boundary_results["min_shape"])
        except Exception as e:
            local_error = e
            error_log = self._log_file(baseline_cache_dir, "error.log")
            error_log.write_text("".join(traceback.format_exception(e)))

        outcome, gathered_results = coordinator.collect_results(result, local_error)
        if backend.is_active:
            backend.deactivate()

        if not outcome.succeeded:
            log(
                "⚠️ %s baseline failed (log: %s), performance comparison unavailable",
                module_type,
                self._log_file(baseline_cache_dir, "error.log"),
                sink=self._sink,
            )
            return

        if result is None:
            raise RuntimeError("Baseline profiling succeeded on all ranks but produced no result")
        result = self._aggregate_result(gathered_results)
        self._baseline_backend = backend
        self._baseline_result = result
        metrics = result.to_json_dict(self._metric_label)
        for metric in ("throughput", "latency"):
            if metric in metrics:
                report_graph_baseline_metric(metric, metrics[metric])
        log("📊 %s baseline: %s", module_type, self._fmt_result(result), sink=self._sink)

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
            self._fmt_result(winner.result),
            sink=self._sink,
        )
        return winner.backend

    def _report_backend_metrics(self, backend_description: str, metrics: dict[str, int | float]) -> None:
        """Record every measured performance metric for a backend."""
        for metric in ("throughput", "latency"):
            if metric in metrics:
                report_backend_metric(metric, backend_description, metrics[metric])

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
        """Profile each candidate at both boundaries and retain the best eligible backend."""
        best: _TuneCandidate | None = None
        profiling_config = self._get_profiling_config(batching, max_batch_size)
        for backend in self._backends:
            built = self._build_and_validate_backend(backend, module, name, graph_spec, samples, device, cache_dir)
            if built is None:
                continue
            try:
                boundaries = self._measure_boundaries(built, name, graph_spec, samples, profiling_config)
            except Exception as error:
                log("❌ Boundary profiling failed for %s: %s", built.describe(), error, sink=self._sink)
                if self.backend_results:
                    self.backend_results[-1]["profiling_error"] = str(error)
                continue
            finally:
                if built.is_active:
                    built.deactivate()
            result = boundaries.get("max_shape", boundaries["min_shape"])
            self._record_boundary_metrics(built, result, boundaries)
            comparison = self._record_perf_result(built, result, boundaries)
            if self._performance_validation_mode is PerformanceValidationMode.ENABLED:
                if comparison is not None and not comparison.passed:
                    continue
            candidate = _TuneCandidate(
                backend=built,
                result=result,
                boundary_results=boundaries,
                speedup=comparison.speedup if comparison is not None else None,
            )
            if best is None or self._is_better_candidate(candidate, best):
                best = candidate
        return best

    def _record_perf_result(
        self,
        backend: Backend,
        result: BackendProfilingResult,
        boundary_results: dict[str, BackendProfilingResult] | None = None,
    ) -> BackendPerfResult | None:
        """Compare profiler means; stability remains the profiler's responsibility."""
        baseline = self._baseline_result
        if baseline is None or baseline.metric <= 0 or result.metric <= 0:
            return None
        boundary_speedups = {}
        if boundary_results:
            for boundary, measured in boundary_results.items():
                boundary_speedups[boundary] = self._speedup(measured, self._baseline_boundary_results[boundary])
        speedup = geometric_mean(boundary_speedups.values()) if boundary_speedups else self._speedup(result, baseline)
        passed = all(value >= self.min_boundary_speedup for value in boundary_speedups.values())
        if not boundary_speedups:
            passed = speedup >= self.min_boundary_speedup
        comparison = BackendPerfResult(
            backend_description=backend.describe(),
            metric=result.metric,
            baseline_metric=baseline.metric,
            speedup=speedup,
            passed=passed,
            boundary_speedups=boundary_speedups,
        )
        self.perf_validation_results.append(comparison)
        if self.backend_results:
            self.backend_results[-1].update({
                "baseline_boundaries": self._boundary_metrics(self._baseline_boundary_results),
                "boundary_speedups": boundary_speedups,
                "speedup": speedup,
                "performance_passed": passed,
            })
        log(
            "📊 %s: boundary speedups=%s, geometric-mean speedup=%.3fx, passed=%s",
            backend.describe(),
            boundary_speedups,
            speedup,
            passed,
            sink=self._sink,
        )
        return comparison

    def _resolve_winner(self, best: _TuneCandidate | None) -> _TuneCandidate:
        """Returns the winning candidate, falling back to the TorchEager baseline when appropriate."""
        use_baseline = (
            self._performance_validation_mode is not PerformanceValidationMode.DISABLED
            and self._baseline_backend is not None
            and self._baseline_result is not None
            and (
                best is None
                or (
                    self._performance_validation_mode is PerformanceValidationMode.ENABLED
                    and (
                        best.speedup <= 1.0
                        if best.speedup is not None
                        else not self._is_better(best.result, self._baseline_result)
                    )
                )
            )
        )
        if use_baseline and self._baseline_backend is not None and self._baseline_result is not None:
            if best is not None and best.backend.is_active:
                best.backend.deactivate()
            reason = (
                "no user backend passed profiling and performance validation"
                if best is None
                else "best user backend did not improve the aggregate boundary score"
            )
            log(
                "ℹ️ Falling back to TorchEager baseline (%s): %s",
                self._fmt_result(self._baseline_result),
                reason,
                sink=self._sink,
            )
            return _TuneCandidate(
                backend=self._baseline_backend,
                result=self._baseline_result,
                boundary_results=self._baseline_boundary_results,
                speedup=1.0,
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
        if backend is None:
            return
        super()._post_tune(backend, name, graph_spec, samples)
        result = next(
            (r for r in self.perf_validation_results if r.backend_description == backend.describe()),
            None,
        )
        if result is None:
            if backend is self._baseline_backend:
                self._log_baseline_selected(backend)
            return
        detail = f"{format(result.baseline_metric, self._value_fmt)} → {self._fmt(result.metric)}"
        if result.boundary_speedups:
            detail = ", ".join(f"{boundary}={speedup:.3f}x" for boundary, speedup in result.boundary_speedups.items())
        msg_short = fmt_speedup_msg(result.speedup, detail, name, backend.describe())
        if self._logger.isEnabledFor(logging.INFO):
            log(msg_short, sink=self._sink)
        else:
            msg_full = f"[AITune] {msg_short}"
            log(msg_full, sink=self._logger.warning)

    def _log_baseline_selected(self, backend: Backend) -> None:
        """Emit an explicit message when the TorchEager baseline is the selected backend."""
        value = f" ({self._fmt_result(self._baseline_result)})" if self._baseline_result is not None else ""
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

    def _describe_parts(self) -> list[str]:
        """Returns the parts of the description."""
        return [
            f"name: {self._title}",
            f"description: {self._description}",
            "backends:",
            *[f"  {backend.describe()}" for backend in self._backends],
        ]

    def _measure_boundaries(
        self,
        backend: Backend,
        name: str,
        graph_spec: GraphSpec,
        samples: SampleStore,
        profiling_config: ProfilingConfig,
    ) -> dict[str, BackendProfilingResult]:
        """Use the configured profiler once per distinct spatial/sequence boundary."""
        boundaries = self._get_shape_boundaries(graph_spec, profiling_config)
        coordinator.verify_equal(boundaries, "profiling shape boundaries")
        results: dict[str, BackendProfilingResult] = {}
        for boundary in boundaries:
            result = self._measure_distributed(
                backend, name, graph_spec, samples, replace(profiling_config, shape_boundary=boundary)
            )
            if result is None or not math.isfinite(result.metric) or result.metric <= 0:
                raise RuntimeError(f"No valid {self._metric_label} for {backend.describe()} at {boundary}")
            results[boundary] = result
            log("📊 %s %s: %s", backend.describe(), boundary, self._fmt_result(result), sink=self._sink)
        return results

    def _get_shape_boundaries(
        self, graph_spec: GraphSpec, profiling_config: ProfilingConfig
    ) -> list[Literal["min_shape", "max_shape"]]:
        # Batch sweeps replace batch axes, so differing batch bounds alone would
        # otherwise repeat the same profiling workload twice.
        for spec in graph_spec.input_spec.tensor_specs:
            batch_axes = spec.get_batch_axis_multipliers() if profiling_config.batching else {}
            for axis, minimum in enumerate(spec.min_shape):
                if axis not in batch_axes and minimum != spec.max_shape[axis]:
                    return ["min_shape", "max_shape"]
        return ["min_shape"]

    def _record_boundary_metrics(
        self, backend: Backend, result: BackendProfilingResult, boundaries: dict[str, BackendProfilingResult]
    ) -> None:
        metrics = result.to_json_dict(self._metric_label)
        if self.backend_results:
            self.backend_results[-1].update(metrics)
            self.backend_results[-1]["shape_boundaries"] = self._boundary_metrics(boundaries)
        self._report_backend_metrics(backend.describe(), metrics)

    def _is_better_candidate(self, candidate: _TuneCandidate, other: _TuneCandidate) -> bool:
        if not candidate.boundary_results or not other.boundary_results:
            return self._is_better(candidate.result, other.result)
        speedups = [
            self._speedup(result, other.boundary_results[boundary])
            for boundary, result in candidate.boundary_results.items()
        ]
        return geometric_mean(speedups) > 1.0

    def _boundary_metrics(self, results: dict[str, BackendProfilingResult]) -> dict[str, dict[str, int | float]]:
        return {boundary: result.to_json_dict(self._metric_label) for boundary, result in results.items()}
