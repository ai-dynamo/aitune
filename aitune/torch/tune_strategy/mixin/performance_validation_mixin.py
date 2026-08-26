# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Performance validation mixin for tune strategy."""

import logging
import shutil
import sys
import traceback
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn as nn

from aitune.torch.backend.backend import Backend
from aitune.torch.backend.torch_eager import TorchEagerBackend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import SampleStore
from aitune.torch.task.find_max_batch_size import find_max_throughput_for_backend
from aitune.torch.task.profiling import (
    ProfilingConfig,
)
from aitune.torch.tune_data.reporting import report_backend_metric, report_graph_baseline_metric
from aitune.torch.tune_strategy.tune_strategy import TuneStrategy
from aitune.utils.logging import log


def fmt_speedup_msg_short(speedup: float, detail: str) -> str:
    """Returns a compact speedup line without module/backend fields."""
    if sys.stdout.isatty():
        lightning = "\033[94m⚡\033[0m"
        speedup_str = f"\033[92m\033[1m{speedup:.2f}x\033[0m"
    else:
        lightning = "⚡"
        speedup_str = f"{speedup:.2f}x"
    return f"{lightning} speedup: {speedup_str} ({detail})"


def fmt_speedup_msg(speedup: float, detail: str, name: str, backend_desc: str) -> str:
    """Returns the full speedup summary line with module/backend fields."""
    if sys.stdout.isatty():
        lightning = "\033[94m⚡\033[0m"
        speedup_str = f"\033[92m\033[1m{speedup:.2f}x\033[0m"
        name_str = f"\033[1m{name}\033[0m"
        backend_str = f"\033[96m{backend_desc}\033[0m"
    else:
        lightning = "⚡"
        speedup_str = f"{speedup:.2f}x"
        name_str = name
        backend_str = backend_desc
    return f"{lightning} {name_str} | backend: {backend_str} | speedup: {speedup_str} ({detail})"


@dataclass
class PerformanceValidationMixinResult:
    """Throughput and speedup result for a single backend."""

    backend_description: str
    throughput: float
    baseline_throughput: float
    speedup: float
    passed: bool


class PerformanceValidationMixin(TuneStrategy):
    """TuneStrategy mixin that validates each backend against a TorchEager throughput baseline.

    When performance validation is enabled, profiles TorchEager during _pre_tune with the strategy profiling
    config narrowed to the resolved batch size (from graph_spec.get_max_batch_size()) to establish a baseline.
    For every candidate backend, profiles with the same task-local profiling config and appends a
    PerformanceValidationMixinResult. Backends with speedup below the configured threshold are rejected.

    Note:
        min_speedup_threshold_percent defaults to 1.0, requiring a measurable 1% speedup when validation is enforced.
        Performance validation is enabled by default.
    """

    def __init__(
        self,
        *args,
        min_speedup_threshold_percent: float = 1.0,
        **kwargs,
    ):
        """Initialize performance validation defaults.

        Args:
            *args: Positional arguments forwarded through cooperative multiple inheritance.
            min_speedup_threshold_percent: Required relative speedup, in percent, when validation is enforced.
                Defaults to 1.0.
            **kwargs: Keyword arguments forwarded through cooperative multiple inheritance.
        """
        super().__init__(*args, **kwargs)
        self.min_speedup_threshold_percent = min_speedup_threshold_percent
        self._performance_validation_enabled: bool = True
        self.perf_validation_results: list[PerformanceValidationMixinResult] = []
        self._baseline_throughput: float | None = None
        self._baseline_backend: Backend | None = None
        self._resolved_batch_size: int | None = None

    def enable_performance_validation(self, enable: bool = True) -> "PerformanceValidationMixin":
        """Enables or disables TorchEager baseline profiling and candidate performance checks."""
        self._performance_validation_enabled = enable
        return self

    def _pre_tune(
        self,
        module: nn.Module,
        name: str,
        graph_spec: GraphSpec,
        samples: SampleStore,
        device: torch.device,
        cache_dir: Path,
    ):
        """Runs pre-tune setup and profiles TorchEager when performance validation is enabled."""
        super()._pre_tune(module, name, graph_spec, samples, device, cache_dir)
        self.perf_validation_results = []
        self._baseline_throughput = None
        self._baseline_backend = None
        self._resolved_batch_size = None

        if not self._performance_validation_enabled:
            log("⚠️ Performance validation against eager baseline is disabled.", sink=self._sink)
            return

        log("🔄 Profiling eager baseline...please wait", sink=self._sink)

        self._resolved_batch_size = graph_spec.get_max_batch_size(normalized=True)

        profiling_cfg = self._profiling_config_for_batch_size(self._resolved_batch_size, batching=True)

        baseline_cache_dir = cache_dir / "perf_validation_baseline"
        shutil.rmtree(baseline_cache_dir, ignore_errors=True)
        baseline_cache_dir.mkdir(parents=True)
        try:
            backend = TorchEagerBackend()
            backend = backend.build(module, graph_spec, samples, device, baseline_cache_dir)
            batch_size, throughput, _ = find_max_throughput_for_backend(
                backend, name, graph_spec, samples, profiling_cfg
            )

            self._baseline_throughput = throughput
            self._baseline_backend = backend
            report_graph_baseline_metric("throughput", throughput)
            log(
                "📊 Eager baseline: batch size=%s, throughput=%.2f samples/s",
                batch_size,
                throughput,
                sink=self._sink,
            )
        except Exception:
            error_log = self._log_file(baseline_cache_dir, "error.log")
            error_log.write_text(traceback.format_exc())
            log(
                "⚠️ Performance validation baseline failed (log: %s), performance check skipped",
                error_log,
                sink=self._sink,
            )

    def _build_validate_and_check_perf(
        self,
        backend: Backend,
        module: nn.Module,
        name: str,
        graph_spec: GraphSpec,
        samples: SampleStore,
        device: torch.device,
        cache_dir: Path,
        *,
        raise_on_failure: bool = False,
    ) -> Backend | None:
        """Build, correctness-validate, then profile and check throughput against baseline.

        Returns the built backend, or None when correctness fails
        (raise_on_failure=False) or performance check rejects it
        (speedup below threshold).
        """
        built = self._build_and_validate_backend(
            backend, module, name, graph_spec, samples, device, cache_dir, raise_on_failure=raise_on_failure
        )
        if built is None:
            return None

        if (
            not self._performance_validation_enabled
            or self._baseline_throughput is None
            or self._resolved_batch_size is None
        ):
            return built

        description = backend.describe()
        profiling_cfg = self._profiling_config_for_batch_size(self._resolved_batch_size, batching=True)

        try:
            batch_size, throughput, _ = find_max_throughput_for_backend(built, name, graph_spec, samples, profiling_cfg)
        except Exception:
            log("⚠️ Performance profiling failed for %s, performance check skipped", description, sink=self._sink)
            return built

        if self._baseline_throughput == 0:
            log("⚠️ Baseline throughput is 0 for %s, performance check skipped", description, sink=self._sink)
            return built

        speedup = throughput / self._baseline_throughput
        passed = speedup >= (1.0 + self.min_speedup_threshold_percent / 100.0)

        report_backend_metric("throughput", description, throughput)
        self.perf_validation_results.append(
            PerformanceValidationMixinResult(
                backend_description=description,
                throughput=throughput,
                baseline_throughput=self._baseline_throughput,
                speedup=speedup,
                passed=passed,
            )
        )

        if sys.stdout.isatty():
            indicator = "\033[92m▲ faster\033[0m" if passed else "\033[33m▼ slower\033[0m"
        else:
            indicator = "▲ faster" if passed else "▼ slower"

        log(
            "📊 %s: batch size=%s, throughput=%.2f samples/s, speedup=%.2fx (%s)",
            description,
            batch_size,
            throughput,
            speedup,
            indicator,
            sink=self._sink,
        )

        if not passed:
            return None

        return built

    def _profiling_config_for_batch_size(self, batch_size: int, *, batching: bool | None = None) -> ProfilingConfig:
        """Returns the strategy profiling config narrowed to one validation batch size."""
        return replace(
            self.profiling_config,
            batch_sizes=[batch_size],
            batching=self.profiling_config.batching if batching is None else batching,
        )

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

        detail = f"{result.baseline_throughput:.2f} → {result.throughput:.2f} samples/s"
        if self._logger.isEnabledFor(logging.INFO):
            log(fmt_speedup_msg_short(result.speedup, detail), sink=self._sink)
        else:
            log(
                f"[AITune] {fmt_speedup_msg(result.speedup, detail, name, backend.describe())}",
                sink=self._logger.warning,
            )

    def _log_baseline_selected(self, backend: Backend) -> None:
        """Emit an explicit message when the TorchEager baseline is the selected backend."""
        throughput = f" ({self._baseline_throughput:.2f} samples/s)" if self._baseline_throughput is not None else ""
        msg = f"ℹ️ Baseline was selected: {backend.describe()}{throughput}"
        if not self._logger.isEnabledFor(logging.INFO):
            return
        log(msg, sink=self._sink)
