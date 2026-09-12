# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""First Wins tune strategy."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from aitune.torch.backend import (
    Backend,
    TensorRTBackend,
    TensorRTBackendConfig,
    TorchInductorAotBackend,
    TorchInductorJitBackend,
)
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import SampleStore
from aitune.torch.tune_strategy.mixin import PerformanceValidationMixin
from aitune.torch.tune_strategy.multi_backend_strategy import MultiBackendStrategy
from aitune.utils.logging import log


class FirstWinsStrategy(PerformanceValidationMixin, MultiBackendStrategy):
    """Try backends in order and stop at the first that passes the configured checks.

    The default order tries TensorRT export paths before Inductor. It is a fallback
    policy, not a measured performance ranking. AOT and JIT use the same order.
    """

    def __init__(self, backends: list[Backend] | None = None, **kwargs):
        """Initializes strategy."""
        super().__init__(backends=backends, **kwargs)

    def to_json_dict(self) -> dict[str, Any]:
        """Returns config dict for first wins strategy."""
        return {
            "backends": [backend.describe() for backend in self._backends],
            "performance_validation_mode": self._performance_validation_mode.value,
            "profiling_config": self._profiling_config_to_json_dict(),
        }

    def _default_aot_backends(self, distributed: bool = False) -> list[Backend]:
        """Try TensorRT before Inductor for AOT; distributed modules need Inductor backends."""
        if distributed:
            return [TorchInductorAotBackend(), TorchInductorJitBackend()]
        return [
            TensorRTBackend(),
            TensorRTBackend(config=TensorRTBackendConfig(use_dynamo=False)),
            TorchInductorJitBackend(),
        ]

    def _default_jit_backends(self, distributed: bool = False) -> list[Backend]:
        """Try TensorRT before Inductor for JIT; distributed modules need Inductor backends."""
        if distributed:
            return [TorchInductorAotBackend(), TorchInductorJitBackend()]
        return [
            TensorRTBackend(),
            TensorRTBackend(config=TensorRTBackendConfig(use_dynamo=False)),
            TorchInductorJitBackend(),
        ]

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

        for backend in self._backends:
            built = self._build_validate_and_check_perf(backend, module, name, graph_spec, samples, device, cache_dir)

            if built is not None:
                log("🎯 Strategy %s execution finished:", self.__class__.__name__, sink=self._sink)
                log("✅ Selected backend: %s", built.describe(), sink=self._sink)
                return built

        raise RuntimeError(f"There is no valid backend for a module: {name}, graph_spec: {graph_spec}")

    def _describe_parts(self) -> list[str]:
        """Returns the parts of the description."""
        return [
            "name: First Wins Strategy",
            "description: evaluate backends in order, return first working backend",
            "backends:",
            *[f"  {backend.describe()}" for backend in self._backends],
        ]
