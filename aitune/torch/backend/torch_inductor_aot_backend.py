# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Torch Inductor AOT backend."""

from collections.abc import Sequence
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, cast

import nvtx
import torch
import torch.nn as nn

from aitune.records import DeploymentArtifact, ModelFiles, RuntimeConfig, TensorSample
from aitune.torch.artifact import artifact_input_samples, bounded_tensor_specs
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
from aitune.torch.module.sample_store import Sample, SampleStore
from aitune.torch.utils.module import move_module_to_device, offload
from aitune.torch.utils.pt2_artifact import PT2CallContract, PT2CallContractState

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
    STATE_GRAPH_SPEC = "graph_spec"
    STATE_PT2_CALL_CONTRACT = "pt2_call_contract"
    STATE_SAMPLES = "samples"

    def __init__(self, config: TorchInductorAotBackendConfig | None = None):
        """Initialize TorchInductorAotBackend.

        Args:
            config: Configuration for AOT Inductor compilation.
        """
        super().__init__()
        self._config = config or TorchInductorAotBackendConfig()
        self._compiled_model_artifact: ArtifactPath | None = None
        self._runner = None
        self._graph_spec: GraphSpec | None = None
        self._pt2_call_contract: PT2CallContract | None = None
        self._pt2_call_contract_error: str | None = None
        self._samples: Sequence[Sample] | None = None

    def key(self) -> str:
        """Returns the key of the backend."""
        return f"{self.__class__.__name__}_{self._config.key()}"

    def describe(self) -> str:
        """Returns the description of the backend."""
        return f"{self.__class__.__name__}({self._config.describe()})"

    def artifact(self) -> DeploymentArtifact:
        """Create the deployment record on request, including after deactivation."""
        try:
            return self._create_artifact()
        except (OSError, RuntimeError, ValueError) as error:
            raise RuntimeError(f"PT2 artifact is not available for deployment: {error}") from error

    def _build(self, module: nn.Module, graph_spec: GraphSpec, samples: SampleStore, cache_dir: Path) -> Backend:
        """Export and compile the model with AOT Inductor, then load the runner."""
        self._graph_spec = graph_spec
        self._samples = samples
        module = module.eval()
        move_module_to_device(module, self._device)

        with self._track_build_step(TorchInductorAotBuildStep.TORCH_EXPORT):
            export_result = TorchExporter().export(module, samples[0], graph_spec, device=self._device)
            exported = export_result.exported_program

        # FIXME: This extra forward pass can mutate model state. Replace it with the shared export contract once the
        # exporter exposes the recorded output structure and tensor ordering.
        self._capture_call_contract(module, export_result.sample)

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

    def _capture_call_contract(self, module: nn.Module, sample: Sample) -> None:
        """Capture PT2 tensor ordering without changing backend build success."""
        self._pt2_call_contract = None
        self._pt2_call_contract_error = None
        try:
            args, kwargs = sample
            with torch.no_grad():
                output = module(*args, **kwargs)
            graph_spec = cast(GraphSpec, self._graph_spec)
            contract = PT2CallContract.capture(graph_spec, sample, output)
            self._pt2_call_contract = contract
        except Exception as error:
            self._pt2_call_contract_error = str(error)
            logger.info("PT2 package will not be available for deployment: %s", error)

    def _activate(self):
        """Load the compiled model from disk."""
        compiled_model_artifact = cast(ArtifactPath, self._compiled_model_artifact)
        logger.debug("Loading compiled AOT Inductor runner from %s.", compiled_model_artifact)
        device_index = self._device.index if self._device.index is not None else 0
        self._runner = torch._inductor.aoti_load_package(str(compiled_model_artifact.path), device_index=device_index)

    def _create_artifact(self) -> DeploymentArtifact:
        """Describe the compiled package using its captured tensor ordering."""
        if self._compiled_model_artifact is None or self._graph_spec is None:
            raise RuntimeError("PT2 artifact requires a compiled package and graph specification")
        if self._pt2_call_contract is None:
            detail = f": {self._pt2_call_contract_error}" if self._pt2_call_contract_error else ""
            raise RuntimeError(f"PT2 artifact requires a captured call contract{detail}")

        contract = self._pt2_call_contract
        input_names = tuple(f"INPUT__{index}" for index in range(len(contract.input_order)))
        output_names = tuple(f"OUTPUT__{index}" for index in range(len(contract.output_order)))
        return DeploymentArtifact(
            model=ModelFiles(
                format="pt2",
                path=self._compiled_model_artifact.path,
                metadata={"structured_call": contract.structured},
            ),
            inputs=bounded_tensor_specs(
                self._graph_spec,
                "input",
                metadata_indices=contract.input_order,
                artifact_names=input_names,
            ),
            outputs=bounded_tensor_specs(
                self._graph_spec,
                "output",
                metadata_indices=contract.output_order,
                artifact_names=output_names,
            ),
            runtime=RuntimeConfig(name="aotinductor"),
            sample_inputs=self._artifact_sample_inputs(contract, input_names),
        )

    def _artifact_sample_inputs(
        self, contract: PT2CallContract, input_names: tuple[str, ...]
    ) -> tuple[tuple[TensorSample, ...], ...]:
        """Derive portable requests from recorded samples in PT2 input order."""
        if self._samples is None:
            return ()
        try:
            return artifact_input_samples(
                cast(GraphSpec, self._graph_spec),
                self._samples,
                metadata_indices=contract.input_order,
                artifact_names=input_names,
            )
        except Exception as error:
            logger.info("Perf Analyzer will use synthetic inputs: %s", error)
            return ()

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

    def to_dict(self) -> dict:
        """Returns the state_dict of the backend."""
        if self._compiled_model_artifact is None or self._graph_spec is None:
            raise RuntimeError("Backend has not been built yet. Please call build() first.")
        return {
            self.STATE_TYPE: self.__class__.__name__,
            self.STATE_COMPILED_MODEL_PATH: self._compiled_model_artifact,
            self.STATE_DEVICE: self._device,
            self.STATE_GRAPH_SPEC: self._graph_spec.to_dict(),
            self.STATE_PT2_CALL_CONTRACT: (
                self._pt2_call_contract.to_dict() if self._pt2_call_contract is not None else None
            ),
            self.STATE_SAMPLES: (self._samples.to_dict() if isinstance(self._samples, SampleStore) else self._samples),
        }

    @classmethod
    def from_dict(cls, module: nn.Module | None, state_dict: dict) -> "TorchInductorAotBackend":
        """Creates a backend from a state_dict."""
        if state_dict.get(cls.STATE_TYPE) != cls.__name__:
            raise ValueError(f"Invalid state_dict type: {state_dict.get(cls.STATE_TYPE)}")

        backend = cls()
        backend._compiled_model_artifact = state_dict[cls.STATE_COMPILED_MODEL_PATH]
        backend._device = state_dict[cls.STATE_DEVICE]
        backend._graph_spec = GraphSpec.from_dict(state_dict[cls.STATE_GRAPH_SPEC])
        contract: PT2CallContractState | None = state_dict[cls.STATE_PT2_CALL_CONTRACT]
        backend._pt2_call_contract = PT2CallContract.from_dict(contract) if contract is not None else None
        samples_state = state_dict.get(cls.STATE_SAMPLES)
        backend._samples = SampleStore.from_dict(samples_state) if isinstance(samples_state, dict) else samples_state
        backend.state = BackendState.CHECKPOINT_LOADED
        return backend
