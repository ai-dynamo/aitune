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
"""First Wins tune strategy."""

from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from aitune.torch.backend import TensorRTBackend, TensorRTBackendConfig, TorchInductorBackend
from aitune.torch.backend.backend import Backend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.tune_strategy.extension import TuneStrategyFindMaxBatchSizeExtension
from aitune.utils.logging import control_output, log
from aitune.utils.timer import Timer


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
            sink=self._sink,
        )

        # Run backend by backend until first working backend is found
        for backend in self._backends:
            backend_cache_dir = cache_dir / backend.key()
            log_file = self._log_file(backend_cache_dir, "build.log")

            with Timer(sink=self._sink, depth=2):
                try:
                    log("⚙️ backend:  %s", backend.describe(), sink=self._sink)
                    log("🔄 in progress...please wait", depth=2, sink=self._sink)
                    with control_output(log_file=log_file):
                        backend = deepcopy(backend)
                        backend = backend.build(module, graph_spec, deepcopy(data), device, backend_cache_dir)
                    log("✅ backend built", depth=2, sink=self._sink)
                    self.check_correctness(backend, name, graph_spec, data)
                    log("✅ backend validated", depth=2, sink=self._sink)
                    selected_backend = backend
                    break
                except Exception:
                    if backend.is_active:
                        backend.deactivate()
                    log("❌ backend failed (log file: %s)", log_file, depth=2, sink=self._sink)
                    module.to(device)  # move module back to device as failed backend could move it to cpu

        if selected_backend:
            log("🎯 Strategy %s execution finished:", self.__class__.__name__, sink=self._sink)
            log("✅ Selected backend: %s", selected_backend.describe(), sink=self._sink)
            return selected_backend

        raise RuntimeError(f"There is no valid backend for a module: {name}, graph_spec: {graph_spec}")

    def _default_backends(self) -> list[Backend]:
        """Returns default backends."""
        return [
            TensorRTBackend(),
            TensorRTBackend(config=TensorRTBackendConfig(use_dynamo=False)),
            TorchInductorBackend(),
        ]

    def _describe_parts(self) -> list[str]:
        """Returns the parts of the description."""
        return [
            "name: First Wins Strategy",
            "description: evaluate backends in order, return first working backend",
            "backends:",
            *[f"  {backend.describe()}" for backend in self._backends],
        ]
