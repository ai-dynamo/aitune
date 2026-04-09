# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base class for tune strategy."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from aitune.torch.backend.backend import Backend, DummyBackend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.task.correctness import check_output_correctness, check_output_tensor_shapes
from aitune.torch.utils.module import count_parameters
from aitune.utils.logging import control_output, log
from aitune.utils.timer import Timer


class TuneStrategy(ABC):
    """Base class for tune strategy."""

    def __init__(self, sink: Callable | None = None):
        """Initializes strategy.

        Args:
            sink: a function where to print status.
            enable_correctness_check: whether to check correctness of the backend.
        """
        self._sink = sink or self._logger.info
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
        with Timer(name=f"Tune `{self.__class__.__name__}`", sink=self._sink):
            self._pre_tune(module, name, graph_spec, data, device, cache_dir)
            backend = self._tune(module, name, graph_spec, data, device, cache_dir)
            self._post_tune(backend, name, graph_spec, data)
            return backend

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
            self._logger.debug(
                "Correctness check is disabled for %s and graph spec %s",
                backend.describe(),
                graph_spec,
            )
            return

        self._logger.debug("Checking correctness for %s and graph spec %s", backend.describe(), graph_spec)
        with torch.no_grad():
            for args, kwargs in data:
                outputs = backend.infer(*deepcopy(args), **deepcopy(kwargs))
                check_output_correctness(outputs, name=f"{name}.{graph_spec.name}.{backend.describe()}.output")
                outputs_metadata = SampleMetadata.from_outputs(outputs)
                check_output_tensor_shapes(graph_spec.output_spec.tensor_specs, outputs_metadata.tensor_specs)

    def enable_correctness_check(self, enable: bool = True) -> "TuneStrategy":
        """Enable/disable correctness checking."""
        self._enable_correctness_check = enable
        return self

    def clone(self) -> "TuneStrategy":
        """Clones the tune strategy."""
        return deepcopy(self)

    def _build_and_validate_backend(
        self,
        backend: Backend,
        module: nn.Module,
        name: str,
        graph_spec: GraphSpec,
        data: list[Sample],
        device: torch.device,
        cache_dir: Path,
        *,
        raise_on_failure: bool = False,
    ) -> Backend | None:
        """Build and validate a backend with standardized error handling and logging.

        This method encapsulates the common workflow shared by all strategies:
        deep-copy the backend, build it, run correctness checks, and handle failures.

        Args:
            backend: Backend to build (will be deep copied).
            module: Module to tune.
            name: Module name.
            graph_spec: Graph specification.
            data: Sample data.
            device: Target device.
            cache_dir: Cache directory for this module/graph.
            raise_on_failure: If True, re-raise the original exception instead of returning None.

        Returns:
            The built and validated backend, or None on failure.
        """
        backend_cache_dir = cache_dir / backend.key()
        log_file = self._log_file(backend_cache_dir, "build.log")

        with Timer(sink=self._sink, depth=2):
            try:
                log("🤖 backend: %s", backend.describe(), sink=self._sink)
                log("🔄 in progress...please wait", depth=2, sink=self._sink)

                with control_output(log_file=log_file):
                    backend = deepcopy(backend)
                    backend = backend.build(module, graph_spec, deepcopy(data), device, backend_cache_dir)

                log("✅ backend built", depth=2, sink=self._sink)
                self.check_correctness(backend, name, graph_spec, data)
                log("✅ backend validated", depth=2, sink=self._sink)

                return backend

            except Exception:
                log("❌ backend failed (log file: %s)", log_file, depth=2, sink=self._sink)
                if raise_on_failure:
                    raise
                if backend.is_active:
                    backend.deactivate()
                module.to(device)  # move module back to device as failed backend could move it to cpu
                return None

    def _pre_tune(
        self,
        module: nn.Module,
        name: str,
        graph_spec: GraphSpec,
        data: list[Sample],
        device: torch.device,
        cache_dir: Path,
    ):
        """Pre-tune hook. Override to add custom logic before tuning."""
        return

    def _post_tune(self, backend: Backend, name: str, graph_spec: GraphSpec, data: list[Sample]):
        """Post-tune hook. Override to add custom logic after tuning."""
        return

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

        Note: each tuning operation should do a deep copy of a backend as tuning could be called multiple times for the
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
        log("number of parameters: %s", count_parameters(module), depth=1, sink=self._sink)
        log("number of layers: %s", self._count_layers(module), depth=1, sink=self._sink)
        log("precisions: %s", precisions, depth=1, sink=self._sink)
        log("graph_spec:", depth=1, sink=self._sink)
        log("input_spec:\n %s", graph_spec.input_spec.describe(), depth=2, sink=self._sink)
        log("output_spec:\n %s", graph_spec.output_spec.describe(), depth=2, sink=self._sink)
        log("num samples: %s", len(data), depth=1, sink=self._sink)
        log("device: %s", device, depth=1, sink=self._sink)
        log("cache_dir: %s", cache_dir, depth=1, sink=self._sink)
        log("strategy:", depth=1, sink=self._sink)
        for part in self._describe_parts():
            log(part, depth=2, sink=self._sink)

    def _log_file(self, cache_dir: Path, filename: str) -> Path:
        cache_dir.mkdir(parents=True, exist_ok=True)
        log_file = cache_dir / filename
        return log_file

    @property
    def _logger(self) -> logging.Logger:
        """Get a logger specific to this backend implementation."""
        return logging.getLogger(f"{self.__class__.__module__}")


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
