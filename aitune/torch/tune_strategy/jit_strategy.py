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
"""Strategy tailored for JIT tuning."""

import copy
from logging import getLogger
from pathlib import Path
from time import perf_counter

import torch
import torch.nn as nn

from aitune.torch.backend.backend import Backend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.tune_strategy.tune_strategy import TuneStrategy
from aitune.utils.logging import control_output, log

logger = getLogger(__name__)


class JitStrategy(TuneStrategy):
    """Strategy tailored for JIT tuning."""

    def __init__(self, backend: Backend):
        """Initializes the JitStrategy with a backend."""
        super().__init__()
        self.backend = backend

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
        start_time = perf_counter()
        selected_backend, backend = None, None

        backend_cache_dir = cache_dir / self.backend.key()
        log_file = self._log_file(backend_cache_dir, "build.log")

        try:
            log("⚙️ backend:  %s", self.backend.describe(), sink=logger.info)
            log("🔄 in progress...please wait", depth=2, sink=logger.info)
            with control_output(log_file=log_file):
                backend = copy.deepcopy(self.backend)
                backend = backend.build(module, graph_spec, data, device, backend_cache_dir)
            log("✅ backend built", depth=2, sink=logger.info)
            self.check_correctness(backend, name, graph_spec, data)
            log("✅ backend validated", depth=2, sink=logger.info)
            selected_backend = backend
        except Exception:
            if backend and backend.is_active:
                backend.deactivate()
            module.to(device)  # move module back to device as failed backend could move it to cpu
            log("❌ backend failed (log file: %s)", log_file, depth=2, sink=logger.info)

        if selected_backend:
            logger.info("🎯 Strategy %s execution finished in %s.", self.__class__.__name__, get_duration(start_time))
            logger.info("✅ Selected backend: %s", selected_backend.describe())
            return selected_backend

        raise RuntimeError(f"There is no valid backend for a module: {name}, graph_spec: {graph_spec}")

    def _describe_parts(self) -> list[str]:
        """Returns the parts of the description."""
        return [
            "name: JIT Strategy",
            "description: tailored for JIT tuning",
            f"backend: {self.backend.describe()}",
        ]


def get_duration(start_time: float) -> str:
    """Returns the duration in a human-readable format.

    Args:
        start_time (float): The start time from perf_counter().

    Returns:
        str: Duration formatted as seconds (e.g., "1.23s") or
             minutes and seconds (e.g., "2m 30s") if duration exceeds 60 seconds.
    """
    duration = perf_counter() - start_time

    if duration >= 60:
        minutes = int(duration // 60)
        seconds = duration % 60
        return f"{minutes}m {seconds:.0f}s"
    else:
        return f"{duration:.2f}s"
