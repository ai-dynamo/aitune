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
"""First Wins tune strategy."""

import copy
from logging import getLogger
from pathlib import Path

import torch
import torch.nn as nn

from aitune.torch.backend import TensorRTBackend, TorchEagerBackend, TorchInductorBackend
from aitune.torch.backend.backend import Backend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.tune_strategy.extension import TuneStrategyFindMaxBatchSizeExtension
from aitune.utils.logging import log

logger = getLogger(__name__)


class FirstWinsStrategy(TuneStrategyFindMaxBatchSizeExtension):
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
        """Tunes given torch module with provided graph_spec and data."""
        selected_backend = None
        log(
            "⏳ Executing strategy `%s` on module `%s` (graph: %s)",
            self.__class__.__name__,
            name,
            graph_spec.name,
            sink=logger.info,
        )
        for backend in self._backends:
            try:
                log("⚙️ backend:  %s", backend.describe(), sink=logger.info)
                log("🔄 in progress...please wait", depth=2, sink=logger.info)
                backend = copy.deepcopy(backend)
                backend = backend.build(module, graph_spec, data, device, cache_dir)
                log("✅ backend built", depth=2, sink=logger.info)
                self.check_correctness(backend, name, graph_spec, data)
                log("✅ backend validated", depth=2, sink=logger.info)
                selected_backend = backend
                break
            except Exception as exception:
                if backend.is_active:
                    backend.deactivate()
                log("❌ backend failed", depth=2, sink=logger.info)
                log("Exception: %s", exception, sink=logger.debug)
                module.to(device)  # move module back to device as failed backend could move it to cpu

        if selected_backend:
            logger.info("🎯 Strategy %s execution finished:", self.__class__.__name__)
            logger.info("✅ Selected backend: %s", selected_backend.describe())
            return selected_backend

        raise RuntimeError(f"There is no valid backend for a module: {name}, graph_spec: {graph_spec}")

    def _default_backends(self) -> list[Backend]:
        """Returns default backends."""
        return [
            TensorRTBackend(),
            TorchInductorBackend(),
            TorchEagerBackend(),
        ]

    def _describe_parts(self) -> list[str]:
        """Returns the parts of the description."""
        return [
            "name: First Wins Strategy",
            "description: evaluate backends in order, return first working backend",
            "backends:",
            *[f"  {backend.describe()}" for backend in self._backends],
        ]
