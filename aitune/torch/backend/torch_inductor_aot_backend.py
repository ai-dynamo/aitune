# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Torch Inductor AOT backend."""

from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, cast

import nvtx
import torch
import torch.nn as nn

from aitune.torch.backend.backend import (
    Backend,
    BackendBuildStep,
    BackendConfig,
    BackendState,
    BuildMode,
    ExecutionMode,
)
from aitune.torch.checkpoint.artifact import ArtifactPath
from aitune.torch.libs.torch import TorchExporter
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import SampleStore
from aitune.torch.utils.module import move_module_to_device, offload

logger = getLogger(__name__)


class TorchInductorAotBuildStep(BackendBuildStep):
    """Identifiers for discrete sub-steps of a TorchInductorAot backend build."""

    TORCH_EXPORT = "Torch export"
    AOT_COMPILE = "AOT compile"


@dataclass
class TorchInductorAotBackendConfig(BackendConfig):
    """Configuration for TorchInductorAotBackend.

    Args:
        inductor_configs (dict): Mapping of ``torch._inductor.config`` attribute names to values,
            passed to ``torch._inductor.aoti_compile_and_package()``.
            Call ``torch._inductor.list_options()`` to see all available keys.
            Example: ``{"max_autotune": True, "coordinate_descent_tuning": True}``
    """

    inductor_configs: dict[str, Any] | None = None


class TorchInductorAotBackend(Backend):
    """Backend that compiles models using AOT Inductor (``torch._inductor.aoti_compile_and_package``).

    Requires PyTorch >= 2.6.

    Performs Ahead-of-Time compilation to a portable ``.pt2`` artifact saved to disk.
    Dynamic batch shapes are inferred automatically from ``graph_spec`` when a batch axis is
    detected. At inference time the artifact is loaded via ``torch._inductor.aoti_load_package``,
    enabling deployment without Python-interpreter overhead.

    Workflow::

        backend = TorchInductorAotBackend()
        # build / tune as usual through ait.tune()
        ait.save(model, "model.ait")
        # later
        ait.load(model, "model.ait")
    """

    _build_mode = BuildMode.AHEAD_OF_TIME
    _execution_modes = frozenset({ExecutionMode.SINGLE_GPU, ExecutionMode.MULTI_GPU})

    # State dictionary keys
    STATE_TYPE = "type"
    STATE_COMPILED_MODEL_PATH = "compiled_model_path"
    STATE_DEVICE = "device"

    def __init__(self, config: TorchInductorAotBackendConfig | None = None):
        """Initialize TorchInductorAotBackend.

        Args:
            config: Configuration for AOT Inductor compilation.
        """
        super().__init__()
        self._config = config or TorchInductorAotBackendConfig()
        self._compiled_model_artifact: ArtifactPath | None = None
        self._runner = None

    def key(self) -> str:
        """Returns the key of the backend."""
        return f"{self.__class__.__name__}_{self._config.key()}"

    def describe(self) -> str:
        """Returns the description of the backend."""
        return f"{self.__class__.__name__}({self._config.describe()})"

    def _build(self, module: nn.Module, graph_spec: GraphSpec, samples: SampleStore, cache_dir: Path) -> Backend:
        """Export and compile the model with AOT Inductor, then load the runner."""
        self._save_config(cache_dir)

        module = module.eval()
        move_module_to_device(module, self._device)

        with self._track_build_step(TorchInductorAotBuildStep.TORCH_EXPORT):
            exported = TorchExporter().export(module, samples[0], graph_spec, device=self._device).exported_program

        with self._track_build_step(TorchInductorAotBuildStep.AOT_COMPILE) as result:
            self._compiled_model_artifact = ArtifactPath(cache_dir, "model.pt2")
            logger.info("Compiling model with AOT Inductor to %s.", self._compiled_model_artifact)
            torch._inductor.aoti_compile_and_package(
                exported,
                package_path=str(self._compiled_model_artifact.path),
                inductor_configs=self._config.inductor_configs or {},
            )
            result["compiled_model_size_bytes"] = self._compiled_model_artifact.path.stat().st_size
        logger.info("AOT Inductor compilation complete with package path %s.", self._compiled_model_artifact)

        # The compiled artifact is self-contained; offload the original module
        # before loading the runner to reduce peak GPU memory.
        offload(module, device="cpu")

        self._activate()
        return self

    def _activate(self):
        """Load the compiled model from disk."""
        compiled_model_artifact = cast(ArtifactPath, self._compiled_model_artifact)
        logger.debug("Loading compiled AOT Inductor runner from %s.", compiled_model_artifact)
        device_index = self._device.index if self._device.index is not None else 0
        self._runner = torch._inductor.aoti_load_package(str(compiled_model_artifact.path), device_index=device_index)

    @nvtx.annotate(message="TorchInductorAotBackend.infer", domain="AITune", color="orange")
    def _infer(self, *args: Any, **kwargs: Any) -> Any:
        """Run inference with the compiled AOT Inductor runner.

        Args:
            *args: Inference arguments.
            **kwargs: Inference keyword arguments.

        Returns:
            Any: The result of the inference.
        """
        with torch.no_grad():
            return self._runner(*args, **kwargs)

    def _deactivate(self):
        """Deactivate backend."""
        self._runner = None

    def _deploy(self):
        """Deploy backend."""
        self._activate()

    def _save_config(self, cache_dir: Path):
        """Store the backend configuration to a file."""
        config_path = cache_dir / "config.json"
        self._config.to_json(config_path)
        logger.info("Config saved to %s", config_path)

    def to_dict(self) -> dict:
        """Returns the state_dict of the backend."""
        if self._compiled_model_artifact is None:
            raise RuntimeError("Backend has not been built yet. Please call build() first.")
        return {
            self.STATE_TYPE: self.__class__.__name__,
            self.STATE_COMPILED_MODEL_PATH: self._compiled_model_artifact,
            self.STATE_DEVICE: self._device,
        }

    @classmethod
    def from_dict(cls, module: nn.Module | None, state_dict: dict) -> "TorchInductorAotBackend":
        """Creates a backend from a state_dict."""
        if state_dict.get(cls.STATE_TYPE) != cls.__name__:
            raise ValueError(f"Invalid state_dict type: {state_dict.get(cls.STATE_TYPE)}")

        backend = cls()
        backend._compiled_model_artifact = state_dict[cls.STATE_COMPILED_MODEL_PATH]
        backend._device = state_dict[cls.STATE_DEVICE]
        backend.state = BackendState.CHECKPOINT_LOADED
        return backend
