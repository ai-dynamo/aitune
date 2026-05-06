# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""First Wins tune strategy."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from aitune.torch.backend import TensorRTBackend, TensorRTBackendConfig, TorchInductorJitBackend
from aitune.torch.backend.backend import Backend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.tune_strategy.mixin import PerformanceValidationMixin
from aitune.utils.logging import log


class FirstWinsStrategy(PerformanceValidationMixin):
    """Strategy which runs backends until it gets first working backend."""

    def __init__(self, backends: list[Backend] | None = None, **kwargs):
        """Initializes strategy."""
        super().__init__(**kwargs)
        self._backends = backends or self._default_backends()

    def _tune(
        self,
        module: nn.Module,
        name: str,
        graph_spec: GraphSpec,
        data: list[Sample],
        device: torch.device,
        cache_dir: Path,
    ) -> Backend:
        """Tunes given torch module witzh provided graph_spec and data."""
        log(
            "⏳ Executing strategy `%s` on module `%s` (graph: %s)",
            self.__class__.__name__,
            name,
            graph_spec.name,
            sink=self._sink,
        )

        for backend in self._backends:
            built = self._build_validate_and_check_perf(backend, module, name, graph_spec, data, device, cache_dir)

            if built is not None:
                log("🎯 Strategy %s execution finished:", self.__class__.__name__, sink=self._sink)
                log("✅ Selected backend: %s", built.describe(), sink=self._sink)
                return built

        raise RuntimeError(f"There is no valid backend for a module: {name}, graph_spec: {graph_spec}")

    def _default_backends(self) -> list[Backend]:
        """Returns default backends."""
        return [
            TensorRTBackend(),
            TensorRTBackend(config=TensorRTBackendConfig(use_dynamo=False)),
            TorchInductorJitBackend(),
        ]

    def _describe_parts(self) -> list[str]:
        """Returns the parts of the description."""
        return [
            "name: First Wins Strategy",
            "description: evaluate backends in order, return first working backend",
            "backends:",
            *[f"  {backend.describe()}" for backend in self._backends],
        ]

    def to_json_dict(self) -> dict[str, Any]:
        """Returns config dict for first wins strategy."""
        return {"backends": [backend.describe() for backend in self._backends]}
