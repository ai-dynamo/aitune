# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Find max batch size mixin for tune strategy.

Looks for best batch size for the module using Torch Eager backend.
"""

import traceback
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from aitune.torch.backend.backend import Backend
from aitune.torch.backend.torch_eager import TorchEagerBackend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.task.find_max_batch_size import find_max_throughput_for_backend
from aitune.torch.tune_strategy.tune_strategy import TuneStrategy
from aitune.utils.logging import control_output


class FindMaxBatchSizeMixin(TuneStrategy):
    """TuneStrategy mixin that finds max batch size.

    Note:
        Find-max-batch-size is enabled by default and uses TorchEagerBackend as the neutral profiling backend.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the mixin defaults."""
        super().__init__(*args, **kwargs)
        self._enable_find_max_batch_size = True
        self._find_max_batch_size_backend_class: type[Backend] = TorchEagerBackend

    def enable_find_max_batch_size(self, enable: bool = True) -> "FindMaxBatchSizeMixin":
        """Enables or disables find max batch size."""
        self._enable_find_max_batch_size = enable
        return self

    def set_find_max_batch_size_default_backend_class(
        self, default_backend_class: type[Backend]
    ) -> "FindMaxBatchSizeMixin":
        """Sets default backend class for find max batch size."""
        self._find_max_batch_size_backend_class = default_backend_class
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
        if self._enable_find_max_batch_size:
            self._logger.info("🚀 Finding max batch size for %s", name)
            find_max_batch_size_cache_dir = cache_dir / "find_max_batch_size"
            build_log_file = self._log_file(find_max_batch_size_cache_dir, "build.log")
            try:
                backend = self._find_max_batch_size_backend_class()
                with control_output(log_file=build_log_file):
                    backend.build(module, graph_spec, deepcopy(data), device, find_max_batch_size_cache_dir)

                max_batch_size, max_throughput, _ = find_max_throughput_for_backend(
                    backend,
                    name,
                    graph_spec,
                    data,
                    self.profiling_config,
                )
                self._logger.info(
                    "✅ Max batch size for %s is %d with throughput %.2f samples/s",
                    name,
                    max_batch_size,
                    max_throughput,
                )
                graph_spec.update_max_batch_size(data[0], max_batch_size)
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
        super()._pre_tune(module, name, graph_spec, data, device, cache_dir)
        self.find_max_batch_size(module, name, graph_spec, data, device, cache_dir)
