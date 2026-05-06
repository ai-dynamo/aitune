# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simple tune strategy."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from aitune.torch.backend.backend import Backend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.tune_strategy.mixin import PerformanceValidationMixin
from aitune.utils.logging import log


class OneBackendStrategy(PerformanceValidationMixin):
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

        built = self._build_validate_and_check_perf(
            self._backend,
            module,
            name,
            graph_spec,
            data,
            device,
            cache_dir,
            raise_on_failure=True,
        )

        if built is None:
            log(
                "⚠️ Backend %s failed performance gate, falling back to TorchEager",
                self._backend.describe(),
                sink=self._sink,
            )
            if self._baseline_backend is None:
                raise RuntimeError(
                    f"Backend '{self._backend.describe()}' failed the performance gate "
                    "but no TorchEager baseline is available to fall back to."
                )
            return self._baseline_backend

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

    def to_json_dict(self) -> dict[str, Any]:
        """Returns config dict for one backend strategy."""
        return {"backend": self._backend.describe()}
