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
"""Simple tune strategy."""

import copy
import logging
from pathlib import Path

import torch
import torch.nn as nn

from aitune.torch.backend.backend import Backend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.tune_strategy.extension import TuneStrategyFindMaxBatchSizeExtension
from aitune.utils.logging import control_output, log

logger = logging.getLogger(__name__)


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
            sink=logger.info,
        )

        backend_cache_dir = cache_dir / self._backend.key()
        log_file = self._log_file(backend_cache_dir, "build.log")

        try:
            log("🤖 backend: %s", self._backend.describe(), sink=logger.info)
            log("🔄 in progress...please wait", depth=2, sink=logger.info)
            with control_output(log_file=log_file):
                backend = copy.deepcopy(self._backend)
                backend = backend.build(module, graph_spec, data, device, backend_cache_dir)
            log("✅ backend built", depth=2, sink=logger.info)
            self.check_correctness(backend, name, graph_spec, data)
            log("✅ backend validated", depth=2, sink=logger.info)
            logger.info("🎯 Strategy %s execution finished:", self.__class__.__name__)
            logger.info("✅ Selected backend: %s", backend.describe())
            return backend
        except Exception as exception:
            log("❌ backend failed (log file: %s)", log_file, depth=2, sink=logger.info)
            raise exception

    def _describe_parts(self):
        """Describes the tuning."""
        return [
            "name: One Backend Strategy",
            "description: Use only one backend",
            f"backend: {self._backend.describe()}",
        ]
