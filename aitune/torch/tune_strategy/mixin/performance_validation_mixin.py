# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Performance validation mixin for tune strategy."""

import logging
import shutil
import sys
import traceback
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn as nn

from aitune.torch.backend.backend import Backend
from aitune.torch.backend.torch_eager import TorchEagerBackend
from aitune.torch.config import (
    DEFAULT_MIN_SPEEDUP_THRESHOLD,
    DEFAULT_STABILITY_PERCENTAGE,
    DEFAULT_THROUGHPUT_BACKOFF_LIMIT,
    DEFAULT_THROUGHPUT_CUTOFF_THRESHOLD,
    DEFAULT_WINDOW_SIZE,
)
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.task.find_max_batch_size import find_max_throughput_for_backend
from aitune.torch.task.profiling import (
    ProfilingConfig,
    StableWindowMeasuringStopStrategy,
    ThroughputSaturatedProfilingStopStrategy,
)
from aitune.torch.tune_data.reporting import report_backend_throughput, report_graph_baseline_throughput
from aitune.torch.tune_strategy.mixin.find_max_batch_size_mixin import FindMaxBatchSizeMixin
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
class PerformanceValidationMixinConfig:
    """Configuration for performance validation.

    profiling_config is an optional user override. When None, a default
    ProfilingConfig is built at runtime for the resolved batch size.
    """

    min_speedup_threshold: float = DEFAULT_MIN_SPEEDUP_THRESHOLD
    profiling_config: ProfilingConfig | None = None

    def profiling_config_for_batch_size(self, batch_size: int) -> ProfilingConfig:
        """Returns profiling config with batch_sizes set to [batch_size]."""
        if self.profiling_config is not None:
            return replace(self.profiling_config, batch_sizes=[batch_size])
        return ProfilingConfig(
            batch_sizes=[batch_size],
            batching=True,
            measurement_stop_strategy=StableWindowMeasuringStopStrategy(
                window_size=DEFAULT_WINDOW_SIZE,
                stability_percentage=DEFAULT_STABILITY_PERCENTAGE,
            ),
            profiling_stop_strategy=ThroughputSaturatedProfilingStopStrategy(
                throughput_cutoff_threshold=DEFAULT_THROUGHPUT_CUTOFF_THRESHOLD,
                throughput_backoff_limit=DEFAULT_THROUGHPUT_BACKOFF_LIMIT,
            ),
        )


@dataclass
class PerformanceValidationMixinResult:
    """Throughput and speedup result for a single backend."""

    backend_description: str
    throughput: float
    baseline_throughput: float
    speedup: float
    passed: bool


class PerformanceValidationMixin(FindMaxBatchSizeMixin):
    """Mixin that validates each correctness-passing backend against a TorchEager throughput baseline.

    Profiles TorchEager during _pre_tune at the resolved batch size (from graph_spec.get_max_batch_size())
    to establish a baseline. For every candidate backend, profiles at the resolved batch size and appends a
    PerformanceValidationMixinResult. When validate_against_baseline=True (default),
    backends with speedup < 1 + min_speedup_threshold are rejected (return None or
    fall back). Profiling and result recording always occur regardless of the flag.
    """

    def __init__(
        self,
        *args,
        perf_validation_config: PerformanceValidationMixinConfig | None = None,
        sink: Callable | None = None,
        **kwargs,
    ):
        """Initializes the mixin."""
        super().__init__(*args, sink=sink, **kwargs)
        self.perf_validation_config = perf_validation_config or PerformanceValidationMixinConfig()
        self.validate_against_baseline: bool = True
        self.perf_validation_results: list[PerformanceValidationMixinResult] = []
        self._baseline_throughput: float | None = None
        self._baseline_backend: Backend | None = None
        self._resolved_batch_size: int | None = None

    def enable_validate_against_baseline(self, enable: bool = True) -> "PerformanceValidationMixin":
        """Enables or disables baseline validation."""
        self.validate_against_baseline = enable
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
        """Calls super()._pre_tune() then profiles TorchEager at the resolved batch size."""
        super()._pre_tune(module, name, graph_spec, data, device, cache_dir)
        self.perf_validation_results = []
        self._baseline_throughput = None
        self._baseline_backend = None

        self._resolved_batch_size = graph_spec.get_max_batch_size()

        profiling_cfg = self.perf_validation_config.profiling_config_for_batch_size(self._resolved_batch_size)

        baseline_cache_dir = cache_dir / "perf_validation_baseline"
        shutil.rmtree(baseline_cache_dir, ignore_errors=True)
        baseline_cache_dir.mkdir(parents=True)
        try:
            backend = TorchEagerBackend()
            backend = backend.build(module, graph_spec, deepcopy(data), device, baseline_cache_dir)

            _, throughput, _ = find_max_throughput_for_backend(backend, name, graph_spec, data, profiling_cfg)
            self._baseline_throughput = throughput
            self._baseline_backend = backend
            report_graph_baseline_throughput(throughput)
            log(
                "📊 TorchEager baseline at bs=%d: %.2f samples/s",
                self._resolved_batch_size,
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
        data: list[Sample],
        device: torch.device,
        cache_dir: Path,
        *,
        raise_on_failure: bool = False,
    ) -> Backend | None:
        """Build, correctness-validate, then profile and check throughput against baseline.

        Returns the built backend, or None when correctness fails
        (raise_on_failure=False) or performance check rejects it
        (validate_against_baseline=True and speedup below threshold).
        """
        built = self._build_and_validate_backend(
            backend, module, name, graph_spec, data, device, cache_dir, raise_on_failure=raise_on_failure
        )
        if built is None:
            return None

        if self._baseline_throughput is None or self._resolved_batch_size is None:
            return built

        description = backend.describe()
        profiling_cfg = self.perf_validation_config.profiling_config_for_batch_size(self._resolved_batch_size)

        try:
            _, throughput, _ = find_max_throughput_for_backend(built, name, graph_spec, data, profiling_cfg)
        except Exception:
            log("⚠️ Performance profiling failed for %s, performance check skipped", description, sink=self._sink)
            return built

        if self._baseline_throughput == 0:
            log("⚠️ Baseline throughput is 0 for %s, performance check skipped", description, sink=self._sink)
            return built

        speedup = throughput / self._baseline_throughput
        passed = speedup >= (1.0 + self.perf_validation_config.min_speedup_threshold)

        report_backend_throughput(description, throughput)
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
            "📊 %s: throughput=%.2f samples/s, speedup=%.2f (%s)",
            description,
            throughput,
            speedup,
            indicator,
            sink=self._sink,
        )

        if not passed and self.validate_against_baseline:
            return None

        return built

    def _post_tune(self, backend: Backend | None, name: str, graph_spec: GraphSpec, data: list[Sample]):
        """Emits a speedup line after tuning completes."""
        if backend is None:
            return
        result = next(
            (r for r in self.perf_validation_results if r.backend_description == backend.describe()),
            None,
        )
        if result is None:
            return
        detail = f"{result.baseline_throughput:.2f} → {result.throughput:.2f} samples/s"
        if self._logger.isEnabledFor(logging.INFO):
            log(fmt_speedup_msg_short(result.speedup, detail), sink=self._sink)
        else:
            log(
                f"[AITune] {fmt_speedup_msg(result.speedup, detail, name, backend.describe())}",
                sink=self._logger.warning,
            )
