# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Max throughput tune strategy.

1. Finds max batch size for Torch Eager as a baseline.
2. Runs all backends with given max batch size.
3. Finds the maximum throughput for each backend.
4. Returns the backend with max throughput.
"""

from dataclasses import dataclass, field
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
from aitune.torch.config import (
    DEFAULT_STABILITY_PERCENTAGE,
    DEFAULT_THROUGHPUT_BACKOFF_LIMIT,
    DEFAULT_THROUGHPUT_CUTOFF_THRESHOLD,
    DEFAULT_WINDOW_SIZE,
)
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.task.find_max_batch_size import find_max_throughput_for_backend
from aitune.torch.task.profiling import (
    MeasuringStopStrategy,
    ModelExecutionTimeMeasuringStrategy,
    ProfilingConfig,
    ProfilingResultEvent,
    ProfilingStopStrategy,
    StableWindowMeasuringStopStrategy,
    ThroughputSaturatedProfilingStopStrategy,
)
from aitune.torch.tune_strategy.extension import TuneStrategyFindMaxBatchSizeExtension
from aitune.utils.logging import log


@dataclass
class MaxThroughputResult:
    """Result of max throughput strategy."""

    max_batch_size: int
    throughput: float
    backend_details: str


@dataclass
class MaxThroughputStrategyResult:
    """Result of max throughput strategy."""

    graph_spec_name: str
    max_throughput_results: list[MaxThroughputResult] = field(default_factory=list)
    measurements: list[ProfilingResultEvent] = field(default_factory=list)


class MaxThroughputStrategy(TuneStrategyFindMaxBatchSizeExtension):
    """Searches and selects the backend with max throughput."""

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
        self._measurement_stop_strategy = measurement_stop_strategy or StableWindowMeasuringStopStrategy(
            window_size=DEFAULT_WINDOW_SIZE,
            stability_percentage=DEFAULT_STABILITY_PERCENTAGE,
        )
        self._profiling_stop_strategy = profiling_stop_strategy or ThroughputSaturatedProfilingStopStrategy(
            throughput_cutoff_threshold=DEFAULT_THROUGHPUT_CUTOFF_THRESHOLD,
            throughput_backoff_limit=DEFAULT_THROUGHPUT_BACKOFF_LIMIT,
        )
        self.results: list[MaxThroughputStrategyResult] = []

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
        measurements = []
        max_throughput_results = []

        # Verify if model has batching
        batching = graph_spec.input_spec.has_batch_axis() and graph_spec.get_max_batch_size() > 1

        # Note: As this strategy is extended with FindMaxBatchSizeTuneStrategyExtension,
        # we can assume that max batch size is included in the graph spec.
        max_batch_size = graph_spec.get_max_batch_size()

        best_backend = None
        best_throughput = float("-inf")
        best_batch_size = 1

        # Run all backends with given max batch size.
        for backend in self._backends:
            backend_cache_dir = cache_dir / backend.key()
            log_file = self._log_file(backend_cache_dir, "build.log")

            built = self._build_and_validate_backend(backend, module, name, graph_spec, data, device, cache_dir)

            if built is None:
                continue

            try:
                batch_size, throughput, results = find_max_throughput_for_backend(
                    built, name, graph_spec, data, self._get_profiling_config(batching, max_batch_size)
                )
                max_throughput_results.append(
                    MaxThroughputResult(
                        max_batch_size=batch_size, throughput=throughput, backend_details=built.describe()
                    )
                )
                measurements += results.entries
                log(
                    "✅ backend profiled - throughput: %.2f samples/s, batch size: %s",
                    throughput,
                    batch_size,
                    depth=2,
                    sink=self._sink,
                )

                if best_backend is None or throughput > best_throughput:
                    log(
                        "🎯 new best throughput for %s is %.2f samples/s, batch size: %s",
                        built.describe(),
                        throughput,
                        batch_size,
                        depth=2,
                        sink=self._sink,
                    )
                    best_backend = built
                    best_throughput = throughput
                    best_batch_size = batch_size

            except Exception:
                if built.is_active:
                    built.deactivate()
                log("❌ backend failed (log file: %s)", log_file, depth=2, sink=self._sink)

        if best_backend is None:
            log(
                "ℹ️ No correct backend found with throughput %f > 0. Backends considered: %s",
                best_throughput,
                ", ".join([backend.describe() for backend in self._backends]),
                sink=self._sink,
            )
            raise RuntimeError("No correct backend found with throughput > 0")

        self.results.append(
            MaxThroughputStrategyResult(
                graph_spec_name=graph_spec.name,
                max_throughput_results=max_throughput_results,
                measurements=measurements,
            )
        )
        best_backend.activate()
        log("🎯 Strategy %s execution finished:", self.__class__.__name__, sink=self._sink)
        log(
            "✅ Selected %s for module %s and graph spec %s.",
            best_backend.describe(),
            name,
            graph_spec,
            sink=self._sink,
        )
        log("   Batch size: %s, throughput: %.2f samples/s", best_batch_size, best_throughput, sink=self._sink)

        return best_backend

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
            TorchEagerBackend(),
        ]

    def _describe_parts(self) -> list[str]:
        """Returns the parts of the description."""
        return [
            "name: Max Throughput Strategy",
            "description: evaluate all backends, return backend with max throughput",
            "backends:",
            *[f"  {backend.describe()}" for backend in self._backends],
        ]
