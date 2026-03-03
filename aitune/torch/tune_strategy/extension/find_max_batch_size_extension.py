# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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
"""Find max batch size extension for tune strategy.

Looks for best batch size for the module using Torch Eager backend.
"""

import traceback
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from aitune.torch.backend.backend import Backend
from aitune.torch.backend.torch_eager import TorchEagerBackend
from aitune.torch.config import (
    DEFAULT_STABILITY_PERCENTAGE,
    DEFAULT_THROUGHPUT_BACKOFF_LIMIT,
    DEFAULT_THROUGHPUT_CUTOFF_THRESHOLD,
    DEFAULT_WINDOW_SIZE,
)
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.task.find_max_batch_size import calculate_highest_throughput_for_backend
from aitune.torch.task.profiling.config import ProfilingConfig
from aitune.torch.task.profiling.measuring_stop_strategy import StableWindowMeasuringStopStrategy
from aitune.torch.task.profiling.measuring_strategy import ModelExecutionTimeMeasuringStrategy
from aitune.torch.task.profiling.profiling_stop_strategy import ThroughputSaturatedProfilingStopStrategy
from aitune.torch.tune_strategy.tune_strategy import TuneStrategy
from aitune.utils.logging import control_output

DEFAULT_MAX_BATCH_SIZE = 2**20


@dataclass
class FindMaxBatchSizeExtensionConfig:
    """Configuration for find max batch size extension."""

    enable_find_max_batch_size: bool = True
    profiling_config: ProfilingConfig | None = None
    default_backend_class: type[Backend] = TorchEagerBackend


class TuneStrategyFindMaxBatchSizeExtension(TuneStrategy):
    """Wrapper for tune strategy that finds max batch size."""

    def __init__(
        self,
        *args,
        sink: Callable | None = None,
        **kwargs,
    ):
        """Initializes wrapper."""
        super().__init__(*args, sink=sink, **kwargs)

        self.find_config = FindMaxBatchSizeExtensionConfig()
        self.find_config.profiling_config = self.default_profiling_config()

    def enable_find_max_batch_size(self, enable: bool = True) -> "TuneStrategyFindMaxBatchSizeExtension":
        """Enables or disables find max batch size."""
        self.find_config.enable_find_max_batch_size = enable
        return self

    def set_find_max_batch_size_profiling_config(
        self, profiling_config: ProfilingConfig
    ) -> "TuneStrategyFindMaxBatchSizeExtension":
        """Sets profiling config for find max batch size."""
        self.find_config.profiling_config = profiling_config
        return self

    def set_find_max_batch_size_default_backend_class(
        self, default_backend_class: type[Backend]
    ) -> "TuneStrategyFindMaxBatchSizeExtension":
        """Sets default backend class for find max batch size."""
        self.find_config.default_backend_class = default_backend_class
        return self

    def find_max_batch_size(
        self,
        module: nn.Module,
        name: str,
        graph_spec: GraphSpec,
        data: list[Sample],
        device: torch.device,
        cache_dir: Path,
    ):
        """Finds max batch size for the module."""
        if self.find_config.enable_find_max_batch_size:
            self._logger.info("🚀 Finding max batch size for %s", name)
            find_max_batch_size_cache_dir = cache_dir / "find_max_batch_size"
            build_log_file = self._log_file(find_max_batch_size_cache_dir, "build.log")
            try:
                backend = self.find_config.default_backend_class()
                with control_output(log_file=build_log_file):
                    backend.build(module, graph_spec, deepcopy(data), device, find_max_batch_size_cache_dir)

                max_batch_size, best_throughput, _ = calculate_highest_throughput_for_backend(
                    backend,
                    name,
                    graph_spec,
                    data,
                    self.find_config.profiling_config,
                )
                self._logger.info(
                    "✅ Max batch size for %s is %d with throughput %.2f samples/s",
                    name,
                    max_batch_size,
                    best_throughput,
                )
                graph_spec.input_spec.update_max_batch_size(data[0], max_batch_size)
            except Exception:
                error_log_file = self._log_file(find_max_batch_size_cache_dir, "error.log")
                error_log_file.write_text(f"Build log file: {build_log_file}\n\nError:\n{traceback.format_exc()}")
                self._logger.info("⚠️ Finding max batch size for `%s` failed (log file: %s)", name, error_log_file)
                raise

    def _pre_tune(
        self,
        module: nn.Module,
        name: str,
        graph_spec: GraphSpec,
        data: list[Sample],
        device: torch.device,
        cache_dir: Path,
    ):
        """Extends tune method to find max batch size."""
        self.find_max_batch_size(module, name, graph_spec, data, device, cache_dir)

    @staticmethod
    def default_profiling_config(
        batching: bool = True,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        window_size: int = DEFAULT_WINDOW_SIZE,
        stability_percentage: float = DEFAULT_STABILITY_PERCENTAGE,
        throughput_cutoff_threshold: float = DEFAULT_THROUGHPUT_CUTOFF_THRESHOLD,
        throughput_backoff_limit: int = DEFAULT_THROUGHPUT_BACKOFF_LIMIT,
    ) -> ProfilingConfig:
        """Get profiling config for finding max batch size.

        Args:
            batching: Whether to profile with batching.
            max_batch_size: Max batch size to find used to construct batch sizes, the batch sizes will be 2^n for n in range(max_batch_size.bit_length()).
            window_size: Window size for measuring stop strategy.
            stability_percentage: Stability percentage for measuring stop strategy.
            throughput_cutoff_threshold: Throughput cutoff threshold for profiling stop strategy.
            throughput_backoff_limit: Throughput backoff limit for profiling stop strategy.

        Returns:
            Profiling config for finding max batch size.

        Note:
            The profiling config will use defaults from highest throughput strategy.
        """
        return ProfilingConfig(
            batching=batching,
            batch_sizes=[2**n for n in range(max_batch_size.bit_length())],
            measuring_strategy=ModelExecutionTimeMeasuringStrategy(),
            measurement_stop_strategy=StableWindowMeasuringStopStrategy(
                window_size=window_size,
                stability_percentage=stability_percentage,
            ),
            profiling_stop_strategy=ThroughputSaturatedProfilingStopStrategy(
                throughput_cutoff_threshold=throughput_cutoff_threshold,
                throughput_backoff_limit=throughput_backoff_limit,
            ),
        )
