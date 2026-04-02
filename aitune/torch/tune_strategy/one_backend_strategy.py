# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simple tune strategy."""

from pathlib import Path

import torch
import torch.nn as nn

from aitune.torch.backend.backend import Backend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.tune_strategy.extension import TuneStrategyFindMaxBatchSizeExtension
from aitune.utils.logging import log


class OneBackendStrategy(TuneStrategyFindMaxBatchSizeExtension):
    """Strategy which uses just one provided backend."""

    def __init__(self, backend: Backend, **kwargs):
        """Initializes strategy."""
        super().__init__(**kwargs)
        self._backend = backend

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

        built = self._build_and_validate_backend(
            self._backend,
            module,
            name,
            graph_spec,
            data,
            device,
            cache_dir,
            raise_on_failure=True,
        )
        assert built is not None

        log("🎯 Strategy %s execution finished:", self.__class__.__name__, sink=self._sink)
        log("✅ Selected backend: %s", built.describe(), sink=self._sink)
        return built

    def _describe_parts(self):
        """Describes the tuning."""
        return [
            "name: One Backend Strategy",
            "description: Use only one backend",
            f"backend: {self._backend.describe()}",
        ]
