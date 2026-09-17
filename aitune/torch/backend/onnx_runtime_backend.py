# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ONNX Runtime backend."""

import copy
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar, cast

import nvtx
import onnx
import onnxruntime
import torch
import torch.nn as nn
from onnx.external_data_helper import _get_all_tensors

from aitune.records import DeploymentArtifact, DType, ModelFiles, RuntimeConfig
from aitune.torch.artifact import bounded_tensor_specs
from aitune.torch.backend.backend import Backend, BackendConfig, BackendState, BuildMode, ExecutionMode, ModuleFormat
from aitune.torch.checkpoint.artifact import ArtifactPath
from aitune.torch.libs.onnx.onnx_exporter import ONNXExporter
from aitune.torch.libs.onnx.runtime import run_onnx
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.onnx_module import OnnxModule
from aitune.torch.module.sample_store import Sample, SampleStore
from aitune.torch.utils.module import offload

logger = getLogger(__name__)


class ONNXExecutionProvider(str, Enum):
    """Supported ONNX Runtime execution providers.

    Only NVIDIA GPU-backed providers are supported:

    * ``CUDA`` — ``CUDAExecutionProvider``: standard GPU execution.
    * ``TENSORRT`` — ``TensorrtExecutionProvider`` with ``CUDAExecutionProvider``
      as fallback: enables TensorRT engine compilation for maximum GPU throughput.
    """

    CUDA = "cuda"
    TENSORRT = "tensorrt"


_ORT_TYPE_TO_DTYPE = {
    "tensor(bool)": DType.BOOL,
    "tensor(uint8)": DType.UINT8,
    "tensor(int8)": DType.INT8,
    "tensor(int16)": DType.INT16,
    "tensor(int32)": DType.INT32,
    "tensor(int64)": DType.INT64,
    "tensor(float16)": DType.FLOAT16,
    "tensor(float)": DType.FLOAT32,
    "tensor(double)": DType.FLOAT64,
}


def _validate_onnx_interface(nodes: list[onnxruntime.NodeArg], specs, kind: str) -> None:
    """Validate the finalized ONNX interface against the recorded Torch contract."""
    for node, spec in zip(nodes, specs, strict=True):
        try:
            dtype = _ORT_TYPE_TO_DTYPE[node.type]
        except KeyError as error:
            raise ValueError(f"ONNX {kind} {node.name!r} uses unsupported element type {node.type!r}") from error
        if dtype is not spec.dtype:
            raise ValueError(f"ONNX {kind} {node.name!r} dtype does not match the recorded graph")
        if node.shape is None or len(node.shape) != len(spec.min_shape):
            raise ValueError(f"ONNX {kind} {node.name!r} rank does not match the recorded graph")
        for axis, dimension in enumerate(node.shape):
            if isinstance(dimension, int) and (spec.min_shape[axis], spec.max_shape[axis]) != (dimension, dimension):
                raise ValueError(
                    f"ONNX {kind} {node.name!r} axis {axis} is fixed at {dimension}, "
                    f"but AITune recorded bounds {spec.min_shape[axis]}..{spec.max_shape[axis]}"
                )


@dataclass
class ONNXRuntimeBackendConfig(BackendConfig):
    """Configuration for ONNXRuntimeBackend.

    Args:
        use_dynamo: If ``True`` (default), export via ``torch.onnx.export(dynamo=True)``
            which calls ``torch.export.export`` internally and produces a more accurate
            graph (no Python-level tracing limitations). If ``False``, use the classic
            trace-based exporter — faster and broader model coverage.
        execution_provider: ONNX Runtime execution provider to use. When ``None``
            the backend defaults to :attr:`ONNXExecutionProvider.CUDA`. Only
            :attr:`ONNXExecutionProvider.CUDA` and
            :attr:`ONNXExecutionProvider.TENSORRT` are supported.
        opset_version: ONNX opset version passed to ``torch.onnx.export``.
            ``None`` uses the torch default.
    """

    use_dynamo: bool = True
    execution_provider: ONNXExecutionProvider | None = None
    opset_version: int | None = None

    def __post_init__(self):
        """Post init."""
        if self.execution_provider is not None:
            try:
                self.execution_provider = ONNXExecutionProvider(self.execution_provider)
            except ValueError as e:
                raise ValueError(
                    f"Invalid execution_provider: {self.execution_provider!r}. "
                    f"Supported values: {[entry.value for entry in list(ONNXExecutionProvider)]}"
                ) from e

    @classmethod
    def from_dict(cls, data: dict) -> "ONNXRuntimeBackendConfig":
        """Initialise config from a plain dict (e.g. parsed from YAML).

        ``execution_provider`` may be passed as a string and will be
        converted to an ``ONNXExecutionProvider`` enum value automatically.
        """
        data = dict(data)
        if data.get("execution_provider") is not None:
            data["execution_provider"] = ONNXExecutionProvider(data["execution_provider"])
        return cls(**data)


class ONNXRuntimeBackend(Backend):
    """Backend that runs Torch or existing ONNX models with ONNX Runtime.

    Uses the source path and shapes directly for ``OnnxModule``. Torch modules
    are exported to ONNX with dynamic dimensions inferred from ``graph_spec``.
    Both paths use the same ONNX Runtime executor.

    Workflow::

        backend = ONNXRuntimeBackend()
        # build / tune as usual through ait.tune()
        ait.save(model, "model.ait")
        # later
        ait.load(model, "model.ait")
    """

    _build_mode = BuildMode.AHEAD_OF_TIME
    _supported_modules = frozenset({ModuleFormat.TORCH, ModuleFormat.ONNX})
    _execution_modes = frozenset({ExecutionMode.SINGLE_GPU})

    # State dictionary keys
    STATE_TYPE = "type"
    STATE_ONNX_MODEL_PATH = "onnx_model_path"
    STATE_ONNX_DATA_PATH = "onnx_data_path"
    STATE_DEVICE = "device"
    STATE_CONFIG = "config"
    STATE_OUTPUT_OBJECT = "output_object"
    STATE_GRAPH_SPEC = "graph_spec"
    STATE_SAMPLES = "samples"
    STATE_EXTERNAL_DATA_PATHS = "external_data_paths"

    _devices: ClassVar[list[str]] = ["cuda"]

    def __init__(self, config: ONNXRuntimeBackendConfig | None = None):
        """Initialize ONNXRuntimeBackend.

        Args:
            config: Configuration for ONNX export and runtime.
        """
        super().__init__()
        self._config = config or ONNXRuntimeBackendConfig()
        self._onnx_model_artifact: ArtifactPath | None = None
        self._onnx_data_artifact: ArtifactPath | None = None
        self._session: onnxruntime.InferenceSession | None = None
        self._output_object = None
        self._external_data_artifacts: list[ArtifactPath] = []
        self._graph_spec: GraphSpec | None = None
        self._samples: SampleStore | None = None
        self._input_nodes: list[onnxruntime.NodeArg] | None = None
        self._output_nodes: list[onnxruntime.NodeArg] | None = None

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
            raise RuntimeError(f"ONNX artifact is not available: {error}") from error

    def _build(self, module: nn.Module, graph_spec: GraphSpec, samples: SampleStore, cache_dir: Path) -> Backend:
        """Use an existing ONNX file or export Torch, then load the session."""
        self._graph_spec = graph_spec

        if isinstance(module, OnnxModule):
            self._use_existing_onnx(module, graph_spec)
        else:
            self._export_onnx(module, graph_spec, samples, cache_dir)
            data_file = Path(str(self._onnx_model_artifact.path) + ".data")
            if data_file.exists():
                self._onnx_data_artifact = ArtifactPath.from_existing(data_file, root=self._onnx_model_artifact.root)

        self._samples = samples
        offload(module, device="cpu")
        self._activate()

        return self

    def _use_existing_onnx(self, module: OnnxModule, graph_spec: GraphSpec) -> None:
        """Prepare artifacts and output structure from an existing ONNX model."""
        self._output_object = {spec.name: None for _, spec in graph_spec.output_spec.tensor_data}
        self._onnx_model_artifact = ArtifactPath.from_existing(module.path, root=module.path.parent)
        # Release the baseline session before allocating the backend's configured session.
        module.deactivate()
        # Read only graph metadata; leave large external weights on disk.
        model = onnx.load(module.path, load_external_data=False)
        locations = {
            entry.value
            for tensor in _get_all_tensors(model)
            for entry in tensor.external_data
            if entry.key == "location"
        }
        self._external_data_artifacts = [
            ArtifactPath.from_existing(module.path.parent / location, root=module.path.parent)
            for location in sorted(locations)
        ]

    def _export_onnx(self, module: nn.Module, graph_spec: GraphSpec, samples: SampleStore, cache_dir: Path) -> None:
        """Export a Torch module and preserve its output structure."""
        self._output_object = self._get_output_object(module=module, sample=samples[0])
        module = module.eval().to(self._device)
        self._onnx_model_artifact = ArtifactPath(cache_dir, "model_raw.onnx")
        onnx_exporter = ONNXExporter(
            output_path=self._onnx_model_artifact.path,
            use_dynamo=self._config.use_dynamo,
            opset_version=self._config.opset_version,
        )
        onnx_exporter.export(module=module, sample=samples[0], graph_spec=graph_spec)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _get_execution_providers(self) -> list[str | tuple]:
        """Return execution providers based on config (no automatic CPU fallback).

        * ``CUDA`` → ``[("CUDAExecutionProvider", {"device_id": ...})]``
        * ``TENSORRT`` → ``["TensorrtExecutionProvider",
          ("CUDAExecutionProvider", {"device_id": ...})]``
        """
        ep = self._config.execution_provider or ONNXExecutionProvider.CUDA
        cuda_ep = ("CUDAExecutionProvider", {"device_id": self._device.index})
        if ep == ONNXExecutionProvider.TENSORRT:
            return ["TensorrtExecutionProvider", cuda_ep]
        return [cuda_ep]

    def _activate(self):
        """Load the ONNX Runtime session from disk."""
        model_artifact = cast(ArtifactPath, self._onnx_model_artifact)
        logger.debug("Loading ONNX Runtime session from %s.", model_artifact)
        providers = self._get_execution_providers()
        self._session = onnxruntime.InferenceSession(str(model_artifact.path), providers=providers)
        if self._samples is not None:
            try:
                self._warmup(self._samples.iter_samples(self._device))
            except Exception:
                self._deactivate()
                raise
        # Keep the finalized interface when the session is released; validate it only on artifact().
        self._input_nodes = self._session.get_inputs()
        self._output_nodes = self._session.get_outputs()

    def _create_artifact(self) -> DeploymentArtifact:
        """Create an artifact from the saved interface and recorded shape bounds."""
        if (
            self._onnx_model_artifact is None
            or self._input_nodes is None
            or self._output_nodes is None
            or self._graph_spec is None
        ):
            raise RuntimeError("ONNX artifact requires a built or deployed backend with a recorded interface")

        model_path = self._onnx_model_artifact.path
        input_nodes = self._input_nodes
        output_nodes = self._output_nodes
        inputs = bounded_tensor_specs(
            self._graph_spec,
            "input",
            recorded_names=tuple(node.name for node in input_nodes),
        )
        outputs = bounded_tensor_specs(
            self._graph_spec,
            "output",
            recorded_names=tuple(node.name for node in output_nodes),
        )
        _validate_onnx_interface(input_nodes, inputs, "input")
        _validate_onnx_interface(output_nodes, outputs, "output")
        additional_files = self._artifact_additional_files(model_path)
        return DeploymentArtifact(
            model=ModelFiles(format="onnx", path=model_path, additional_files=additional_files),
            inputs=inputs,
            outputs=outputs,
            runtime=RuntimeConfig(
                name="onnxruntime",
                options={"execution_provider": (self._config.execution_provider or ONNXExecutionProvider.CUDA).value},
            ),
        )

    def _artifact_additional_files(self, model_path: Path) -> tuple[Path, ...]:
        """Return every tracked ONNX data file once, relative to the model."""
        artifacts = list(self._external_data_artifacts)
        if self._onnx_data_artifact is not None:
            artifacts.append(self._onnx_data_artifact)
        relative_paths = (artifact.path.relative_to(model_path.parent) for artifact in artifacts)
        return tuple(dict.fromkeys(relative_paths))

    def _warmup(self, samples: Iterable[Sample]) -> None:
        """Run representative samples to initialize the execution provider.

        Some execution providers defer work until inference. In particular, the TensorRT
        execution provider compiles the ONNX graph when it first runs. Exercising a small
        number of recorded samples makes those failures part of the backend build instead
        of deferring them until profiling or application inference.

        Args:
            samples: Recorded input samples for the backend build.
        """
        logger.info("Warming up ONNX Runtime execution provider.")
        for args, kwargs in samples:
            self._infer(*args, **kwargs)

    def _prepare_inputs(self, args: tuple, kwargs: dict) -> dict[str, Any]:
        """Map args/kwargs to session input names using graph_spec locators.

        Tensors are returned as-is (preserving their device); I/O binding happens
        in the shared ``run_onnx`` executor.
        """
        session_input_names = {inp.name for inp in self._session.get_inputs()}
        inputs: dict[str, Any] = {}
        forward_inputs = self._graph_spec.forward_signature.normalize(args, kwargs)
        for locator, tensor_spec in self._graph_spec.input_spec.tensor_data:
            name = GraphSpec.tensor_name(locator, tensor_spec, "input")
            if name not in session_input_names:
                logger.debug("Input: %s not found in session inputs", name)
                continue
            inputs[name] = locator.get_value(forward_inputs.arguments)
        return inputs

    def _prepare_outputs(self, outputs: dict[str, torch.Tensor]) -> Any:
        """Reconstruct original output structure from session output tensors."""
        result = copy.deepcopy(self._output_object)
        for locator, tensor_spec in self._graph_spec.output_spec.tensor_data:
            name = GraphSpec.tensor_name(locator, tensor_spec, "output")
            if name in outputs:
                result = locator.set_value(result, outputs[name])
            else:
                logger.debug("Output: %s not found in session outputs", name)
        return result

    @nvtx.annotate(message="ONNXRuntimeBackend.infer", domain="AITune", color="green")
    def _infer(self, *args: Any, **kwargs: Any) -> Any:
        """Run inference through the shared ONNX Runtime I/O binding executor."""
        inputs = self._prepare_inputs(args, kwargs)
        return self._prepare_outputs(run_onnx(self._session, inputs, self._device))

    def _get_output_object(self, module: nn.Module, sample: Sample) -> Any:
        """Get the output object from the module and sample.

        Args:
            module: PyTorch module
            sample: Sample input to use for model inference.

        Returns:
            The output object from the module.

        Note: to avoid case where a module returns a reference to the input argument, we make a deep copy of
        the output object.
        """
        module.to(self._device)
        args, kwargs = sample
        with torch.no_grad():
            output_object = module(*args, **kwargs)
        return copy.deepcopy(output_object)

    def _deactivate(self):
        """Deactivate backend."""
        self._session = None

    def _deploy(self):
        """Deploy backend."""
        self._activate()
        self._samples = None

    def to_dict(self) -> dict:
        """Returns the state_dict of the backend."""
        if self._onnx_model_artifact is None:
            raise RuntimeError("Backend has not been built yet. Please call build() first.")
        state = {
            self.STATE_TYPE: self.__class__.__name__,
            self.STATE_EXTERNAL_DATA_PATHS: self._external_data_artifacts,
            self.STATE_CONFIG: self._config.to_dict(),
            self.STATE_ONNX_MODEL_PATH: self._onnx_model_artifact,
            self.STATE_OUTPUT_OBJECT: self._output_object,
            self.STATE_GRAPH_SPEC: self._graph_spec.to_dict(),
            self.STATE_DEVICE: self._device,
            self.STATE_SAMPLES: self._samples.to_dict() if self._samples is not None else None,
        }
        if self._onnx_data_artifact is not None:
            state[self.STATE_ONNX_DATA_PATH] = self._onnx_data_artifact
        return state

    @classmethod
    def from_dict(cls, module: nn.Module | None, state_dict: dict) -> "ONNXRuntimeBackend":
        """Creates a backend from a state_dict."""
        if state_dict.get(cls.STATE_TYPE) != cls.__name__:
            raise ValueError(f"Invalid state_dict type: {state_dict.get(cls.STATE_TYPE)}")

        config = ONNXRuntimeBackendConfig.from_dict(state_dict[cls.STATE_CONFIG])

        backend = cls(config=config)
        backend._external_data_artifacts = state_dict.get(cls.STATE_EXTERNAL_DATA_PATHS, [])
        backend._onnx_model_artifact = state_dict[cls.STATE_ONNX_MODEL_PATH]
        backend._onnx_data_artifact = state_dict.get(cls.STATE_ONNX_DATA_PATH)
        backend._device = state_dict[cls.STATE_DEVICE]
        backend._graph_spec = GraphSpec.from_dict(state_dict[cls.STATE_GRAPH_SPEC])
        backend._output_object = state_dict[cls.STATE_OUTPUT_OBJECT]
        samples_state = state_dict.get(cls.STATE_SAMPLES)
        backend._samples = SampleStore.from_dict(samples_state) if samples_state is not None else None
        backend.state = BackendState.CHECKPOINT_LOADED
        return backend
