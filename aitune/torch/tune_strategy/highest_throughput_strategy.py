# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Highest throughput tune strategy.

1. Finds max batch size for Torch Eager as a baseline.
2. Runs all backends with given max batch size.
3. Finds the highest throughput for each backend.
4. Returns the backend with the highest throughput.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from aitune.torch.backend import (
    Backend,
    TensorRTBackend,
    TorchEagerBackend,
    TorchInductorBackend,
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
from aitune.torch.task.find_max_batch_size import calculate_highest_throughput_for_backend
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
from aitune.utils.logging import control_output, log
from aitune.utils.timer import Timer


@dataclass
class HighestThroughputResult:
    """Result of highest throughput strategy."""

    max_batch_size: int
    throughput: float
    backend_details: str


@dataclass
class HighestThroughputStrategyResult:
    """Result of highest throughput strategy."""

    graph_spec_name: str
    highest_throughput_results: list[HighestThroughputResult] = field(default_factory=list)
    measurements: list[ProfilingResultEvent] = field(default_factory=list)


class HighestThroughputStrategy(TuneStrategyFindMaxBatchSizeExtension):
    """Searches and selects the backend with the highest throughput."""

    def __init__(
        self,
        backends: list[Backend] | None = None,
        measurement_stop_strategy: MeasuringStopStrategy | None = None,
        profiling_stop_strategy: ProfilingStopStrategy | None = None,
        **kwargs,
    ):
        """Initializes strategy.

        Args:
            backends: List of backends to tune.
            measurement_stop_strategy: Measurement stop strategy.
            profiling_stop_strategy: Profiling stop strategy.
            **kwargs: Additional arguments for the parent class
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
        self.results: list[HighestThroughputStrategyResult] = []

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
        highest_throughput_results = []

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

            with Timer(sink=self._sink, depth=2):
                try:
                    log("🤖 backend: %s", backend.describe(), sink=self._sink)
                    log("🔄 in progress...please wait", depth=2, sink=self._sink)
                    with control_output(log_file=log_file):
                        backend = deepcopy(backend)
                        backend.build(module, graph_spec, deepcopy(data), device, backend_cache_dir)
                    log("✅ backend built", depth=2, sink=self._sink)
                    self.check_correctness(backend, name, graph_spec, data)
                    log("✅ backend validated", depth=2, sink=self._sink)
                    batch_size, throughput, results = calculate_highest_throughput_for_backend(
                        backend, name, graph_spec, data, self._get_profiling_config(batching, max_batch_size)
                    )
                    highest_throughput_results.append(
                        HighestThroughputResult(
                            max_batch_size=batch_size, throughput=throughput, backend_details=backend.describe()
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
                            backend.describe(),
                            throughput,
                            batch_size,
                            depth=2,
                            sink=self._sink,
                        )
                        best_backend = backend
                        best_throughput = throughput
                        best_batch_size = batch_size

                except Exception:
                    if backend.is_active:
                        backend.deactivate()
                    log("❌ backend failed (log file: %s)", log_file, depth=2, sink=self._sink)
                    module.to(device)  # move module back to device as failed backend could move it to cpu

        if best_backend is None:
            log(
                "ℹ️ No correct backend found with throughput %f > 0. Backends considered: %s",
                best_throughput,
                ", ".join([backend.describe() for backend in self._backends]),
                sink=self._sink,
            )
            raise RuntimeError("No correct backend found with throughput > 0")

        self.results.append(
            HighestThroughputStrategyResult(
                graph_spec_name=graph_spec.name,
                highest_throughput_results=highest_throughput_results,
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
            TorchInductorBackend(),
            TorchTensorRTJitBackend(),
            TorchTensorRTAotBackend(),
            TorchEagerBackend(),
        ]

    def _describe_parts(self) -> list[str]:
        """Returns the parts of the description."""
        return [
            "name: Highest Throughput Strategy",
            "description: evaluate all backends, return backend with highest throughput",
            "backends:",
            *[f"  {backend.describe()}" for backend in self._backends],
        ]
