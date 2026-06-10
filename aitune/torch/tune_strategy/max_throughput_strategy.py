# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Max throughput tune strategy.

1. Finds max batch size for Torch Eager as a baseline.
2. Profiles TorchEager at that batch size sweep as a throughput baseline.
3. Runs all user-provided backends with the same sweep.
4. Returns the backend with max throughput; falls back to TorchEager when
   validate_against_baseline is enabled and no user backend beats the baseline.
"""

import logging
import shutil
import traceback
from copy import deepcopy
from dataclasses import dataclass
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
from aitune.torch.task.find_max_batch_size import find_max_throughput_for_backend
from aitune.torch.task.profiling import (
    MeasuringStopStrategy,
    ModelExecutionTimeMeasuringStrategy,
    NumStepsMeasuringStopStrategy,
    ProfilingConfig,
    ProfilingStopStrategy,
    ThroughputSaturatedProfilingStopStrategy,
)
from aitune.torch.tune_data.reporting import report_backend_throughput, report_graph_baseline_throughput
from aitune.torch.tune_strategy.mixin import FindMaxBatchSizeMixin
from aitune.torch.tune_strategy.mixin.performance_validation_mixin import (
    PerformanceValidationMixinResult,
    fmt_speedup_msg,
)
from aitune.utils.logging import log


@dataclass
class _TuneCandidate:
    backend: Backend
    throughput: float
    batch_size: int


class MaxThroughputStrategy(FindMaxBatchSizeMixin):
    """Searches and selects the backend with max throughput.

    TorchEager is profiled in _pre_tune as a throughput baseline (not injected into
    the backends list). When validate_against_baseline is enabled (default), the strategy
    falls back to TorchEager when no user-provided backend beats it. When disabled,
    the best user-provided backend wins regardless of speed, and the strategy raises
    if all user backends fail.
    """

    def __init__(
        self,
        backends: list[Backend] | None = None,
        measurement_stop_strategy: MeasuringStopStrategy | None = None,
        profiling_stop_strategy: ProfilingStopStrategy | None = None,
        **kwargs: Any,
    ):
        """Initializes strategy.

        Args:
            backends: List of backends to tune.
            measurement_stop_strategy: Measurement stop strategy.
            profiling_stop_strategy: Profiling stop strategy.
            kwargs: Additional arguments for the parent class
        """
        super().__init__(**kwargs)
        self._backends = backends or self._default_backends()
        self._validate_against_baseline: bool = True
        self._measurement_stop_strategy = measurement_stop_strategy or NumStepsMeasuringStopStrategy()
        self._profiling_stop_strategy = profiling_stop_strategy or ThroughputSaturatedProfilingStopStrategy()

        self.perf_validation_results: list[PerformanceValidationMixinResult] = []
        self._baseline_throughput: float | None = None
        self._baseline_backend: Backend | None = None
        self._baseline_batch_size: int | None = None

    def enable_validate_against_baseline(self, enable: bool = True) -> "MaxThroughputStrategy":
        """Enables or disables baseline validation."""
        self._validate_against_baseline = enable
        return self

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
        self._baseline_throughput = None
        self._baseline_backend = None
        self._baseline_batch_size = None

        batching = graph_spec.input_spec.has_batch_axis() and graph_spec.get_max_batch_size() > 1
        max_batch_size = graph_spec.get_max_batch_size()
        profiling_cfg = self._get_profiling_config(batching, max_batch_size)

        baseline_cache_dir = cache_dir / "baseline"
        shutil.rmtree(baseline_cache_dir, ignore_errors=True)
        baseline_cache_dir.mkdir(parents=True)
        try:
            backend = TorchEagerBackend()
            backend = backend.build(module, graph_spec, deepcopy(data), device, baseline_cache_dir)
            batch_size, throughput, _ = find_max_throughput_for_backend(backend, name, graph_spec, data, profiling_cfg)
            self._baseline_throughput = throughput
            self._baseline_backend = backend
            self._baseline_batch_size = batch_size
            report_graph_baseline_throughput(throughput)
            log("📊 TorchEager baseline: %.2f samples/s", throughput, sink=self._sink)
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
        log("   Batch size: %s, throughput: %.2f samples/s", winner.batch_size, winner.throughput, sink=self._sink)
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
                batch_size, throughput, _ = find_max_throughput_for_backend(
                    built, name, graph_spec, data, self._get_profiling_config(batching, max_batch_size)
                )
                self.backend_results[-1].update(throughput=throughput, max_batch_size=batch_size)
                report_backend_throughput(built.describe(), throughput)
                log(
                    "✅ backend profiled - throughput: %.2f samples/s, batch size: %s",
                    throughput,
                    batch_size,
                    depth=2,
                    sink=self._sink,
                )
                self._record_perf_result(built, throughput)
                if best is None or throughput > best.throughput:
                    log(
                        "🎯 new best throughput for %s is %.2f samples/s, batch size: %s",
                        built.describe(),
                        throughput,
                        batch_size,
                        depth=2,
                        sink=self._sink,
                    )
                    best = _TuneCandidate(backend=built, throughput=throughput, batch_size=batch_size)
            except Exception:
                if built.is_active:
                    built.deactivate()
                log("❌ backend failed (log file: %s)", log_file, depth=2, sink=self._sink)

        return best

    def _record_perf_result(self, backend: Backend, throughput: float) -> None:
        """Appends a PerformanceValidationMixinResult for the given backend if a baseline is available."""
        if self._baseline_throughput is None or self._baseline_throughput <= 0:
            return
        speedup = throughput / self._baseline_throughput
        self.perf_validation_results.append(
            PerformanceValidationMixinResult(
                backend_description=backend.describe(),
                throughput=throughput,
                baseline_throughput=self._baseline_throughput,
                speedup=speedup,
                passed=speedup >= 1.0,
            )
        )

    def _resolve_winner(self, best: _TuneCandidate | None) -> _TuneCandidate:
        """Returns the winning candidate, falling back to the TorchEager baseline when appropriate."""
        use_baseline = (
            self._validate_against_baseline
            and self._baseline_backend is not None
            and self._baseline_throughput is not None
            and (best is None or best.throughput < self._baseline_throughput)
        )
        if use_baseline:
            reason = (
                "no user backend succeeded"
                if best is None
                else f"best user backend ({best.throughput:.2f} samples/s) slower than baseline"
            )
            log(
                "ℹ️ Falling back to TorchEager baseline (%.2f samples/s): %s",
                self._baseline_throughput,
                reason,
                sink=self._sink,
            )
            return _TuneCandidate(
                backend=self._baseline_backend,
                throughput=self._baseline_throughput,
                batch_size=self._baseline_batch_size or 1,
            )
        if best is None:
            log(
                "ℹ️ No correct backend found with throughput > 0. Backends considered: %s",
                ", ".join([b.describe() for b in self._backends]),
                sink=self._sink,
            )
            raise RuntimeError("No correct backend found with throughput > 0")
        return best

    def _post_tune(self, backend: Backend | None, name: str, graph_spec: GraphSpec, data: list[Sample]):
        """Emits a speedup line after tuning completes."""
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
        detail = f"{result.baseline_throughput:.2f} → {result.throughput:.2f} samples/s"
        msg_short = fmt_speedup_msg(result.speedup, detail, name, backend.describe())
        if self._logger.isEnabledFor(logging.INFO):
            log(msg_short, sink=self._sink)
        else:
            msg_full = f"[AITune] {msg_short}"
            log(msg_full, sink=self._logger.warning)

    def _log_baseline_selected(self, backend: Backend) -> None:
        """Emit an explicit message when the TorchEager baseline is the selected backend."""
        throughput = f" ({self._baseline_throughput:.2f} samples/s)" if self._baseline_throughput is not None else ""
        msg = f"ℹ️ Baseline was selected: {backend.describe()}{throughput}"
        if not self._logger.isEnabledFor(logging.INFO):
            return
        log(msg, sink=self._sink)

    def _get_profiling_config(self, batching: bool, max_batch_size: int) -> ProfilingConfig:
        """Gets profiling configuration."""
        return ProfilingConfig(
            batching=batching,
            batch_sizes=[2**n for n in range(max_batch_size.bit_length())],
            measuring_strategy=ModelExecutionTimeMeasuringStrategy(),
            measurement_stop_strategy=self._measurement_stop_strategy,
            profiling_stop_strategy=self._profiling_stop_strategy,
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
            "name: Max Throughput Strategy",
            "description: evaluate all backends, return backend with max throughput",
            "backends:",
            *[f"  {backend.describe()}" for backend in self._backends],
        ]

    def to_json_dict(self) -> dict[str, Any]:
        """Returns config dict for max throughput strategy."""
        return {
            "backends": [b.describe() for b in self._backends],
            "measurement_stop_strategy": self._measurement_stop_strategy.__class__.__name__,
            "profiling_stop_strategy": self._profiling_stop_strategy.__class__.__name__,
        }
