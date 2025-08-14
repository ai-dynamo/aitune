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
"""Base class for tune strategy."""

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from logging import getLogger
from pathlib import Path

import torch
import torch.nn as nn

from aitune.torch.backend.backend import Backend, DummyBackend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import OUTPUT_METADATA_PREFIX, Sample
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.task.correctness import check_output_correctness, check_output_tensor_shapes
from aitune.utils.logging import log

logger = getLogger(__name__)


class TuneStrategy(ABC):
    """Base class for tune strategy."""

    def __init__(self, sink: Callable = logger.info):
        """Initializes strategy.

        Args:
            sink: a function where to print status.
            enable_correctness_check: whether to check correctness of the backend.
        """
        self._sink = sink
        self._enable_correctness_check = True

    def tune_dry_run(
        self,
        module: nn.Module,
        name: str,
        graph_spec: GraphSpec,
        data: list[Sample],
        device: torch.device,
        cache_dir: Path,
    ):
        """Performs tune dry run."""
        self._describe(module, name, graph_spec, data, device, cache_dir, dry_run=True)

    def describe(self) -> str:
        """Describes what strategy is doing."""
        return "\n".join(self._describe_parts())

    def tune(
        self,
        module: nn.Module,
        name: str,
        graph_spec: GraphSpec,
        data: list[Sample],
        device: torch.device,
        cache_dir: Path,
    ) -> Backend:
        """Tunes given torch module with provided graph_spec and data."""
        self._describe(module, name, graph_spec, data, device, cache_dir)
        return self._tune(module, name, graph_spec, data, device, cache_dir)

    def check_correctness(self, backend: Backend, name: str, graph_spec: GraphSpec, data: list[Sample]):
        """Check outputs for NaN/inf.

        Args:
            backend: The backend to check.
            name: The name of the module.
            graph_spec: The graph spec of the module.
            data: The data to check.

        Note:
            This method is should be called by the _tune method to check the correctness of the backend.

            You can disable correctness check by calling `enable_correctness_check(False)`.

        Raises:
            CorrectnessCheckError: if the backend fails any check.
        """
        if not self._enable_correctness_check:
            logger.debug(
                "Correctness check is disabled for %s and graph spec %s",
                backend.describe(),
                graph_spec,
            )
            return

        logger.debug("Checking correctness for %s and graph spec %s", backend.describe(), graph_spec)
        with torch.inference_mode():
            for args, kwargs in data:
                outputs = backend.infer(*args, **kwargs)
                check_output_correctness(outputs, name=f"{name}.{graph_spec.name}.{backend.describe()}.output")
                outputs_metadata = SampleMetadata.from_sample(outputs, prefix=OUTPUT_METADATA_PREFIX)
                check_output_tensor_shapes(graph_spec.output_spec.tensor_specs, outputs_metadata.tensor_specs)

    def enable_correctness_check(self, enable: bool = True) -> "TuneStrategy":
        """Enable/disable correctness checking."""
        self._enable_correctness_check = enable
        return self

    def clone(self) -> "TuneStrategy":
        """Clones the tune strategy."""
        return copy.deepcopy(self)

    @abstractmethod
    def _tune(
        self,
        module: nn.Module,
        name: str,
        graph_spec: GraphSpec,
        data: list[Sample],
        device: torch.device,
        cache_dir: Path,
    ) -> Backend:
        """Tunes given torch module with provided graph_spec and data.

        Note: each tuning operation should do a deepcopy of a backend as tuning could be called multiple times for the
        same module i.e. if there are different graph specs

        Returns:
            The tuned and activated backend.

        Raises:
            RuntimeError: if the backend fails any check.
        """
        ...

    @abstractmethod
    def _describe_parts(self) -> list[str]:
        """Returns the parts of the description."""
        ...

    @staticmethod
    def _count_parameters(module: nn.Module) -> str:
        """Counts the total number of parameters and returns it in a human-readable format (e.g., 1.2M, 500K)."""
        num_params = sum(p.numel() for p in module.parameters())

        if num_params >= 1_000_000_000:
            return f"{num_params / 1_000_000_000:.1f}B"
        elif num_params >= 1_000_000:
            return f"{num_params / 1_000_000:.1f}M"
        elif num_params >= 1_000:
            return f"{num_params / 1_000:.1f}K"
        else:
            return f"{num_params}"

    @staticmethod
    def _count_layers(module: nn.Module) -> int:
        return len(list(module.named_children()))

    @staticmethod
    def _layers_precisions(module: nn.Module) -> set:
        """Get the layers precisions of the module and return a set of unique precisions.

        Returns:
            Set of unique precisions
        """
        layer_precisions = set()
        for _, m in module.named_modules():
            if list(m.parameters()):  # Only check modules with parameters
                # Get the dtype of the first parameter (they should all be the same)
                param_dtype = next(m.parameters()).dtype
                layer_precisions.add(param_dtype)

        return layer_precisions

    def _describe(
        self,
        module: nn.Module,
        name: str,
        graph_spec: GraphSpec,
        data: list[Sample],
        device: torch.device,
        cache_dir: Path,
        dry_run: bool = False,
    ):
        """Describes the tune strategy."""
        precisions = ", ".join(str(p) for p in self._layers_precisions(module))
        log("------------------------------------------------------------", sink=self._sink)
        log(
            "🚀 Tuning graph `%s` for module `%s`" + (" (DRY RUN):" if dry_run else ":"),
            graph_spec.name,
            name,
            sink=self._sink,
        )
        log("number of parameters: %s", self._count_parameters(module), depth=1, sink=self._sink)
        log("number of layers: %s", self._count_layers(module), depth=1, sink=self._sink)
        log("precisions: %s", precisions, depth=1, sink=self._sink)
        log("graph_spec:", depth=1, sink=self._sink)
        log("input_spec: %s", graph_spec.input_spec.describe(), depth=2, sink=self._sink)
        log("output_spec: %s", graph_spec.output_spec.describe(), depth=2, sink=self._sink)
        log("num samples: %s", len(data), depth=1, sink=self._sink)
        log("device: %s", device, depth=1, sink=self._sink)
        log("cache_dir: %s", cache_dir, depth=1, sink=self._sink)
        log("strategy:", depth=1, sink=self._sink)
        for part in self._describe_parts():
            log(part, depth=2, sink=self._sink)


class DummyTuneStrategy(TuneStrategy):
    """Dummy tune strategy that does nothing."""

    def _describe_parts(self):
        """Describes what strategy is doing."""
        return ["Dummy strategy which does nothing."]

    def _tune(
        self,
        module: nn.Module,
        name: str,
        graph_spec: GraphSpec,
        data: list[Sample],
        device: torch.device,
        cache_dir: Path,
    ):
        """Tunes given torch module with provided graph_spec and data."""
        return DummyBackend()
