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

import numpy as np
import nvtx
import onnxruntime
import torch
import torch.nn as nn

from aitune.torch.backend.backend import Backend, BackendConfig, BackendState
from aitune.torch.checkpoint.artifact import ArtifactPath
from aitune.torch.libs.cuda.memory import memcpy_to_torch
from aitune.torch.libs.onnx.onnx_exporter import ONNXExporter
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import Sample, SampleStore
from aitune.torch.utils.module import offload
from aitune.torch.utils.tensor import format_tensor_name

logger = getLogger(__name__)


# Mapping from torch dtype to numpy dtype for ONNX Runtime IOBinding.
_TORCH_DTYPE_TO_NUMPY: dict[torch.dtype, type] = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,
}


class ONNXExecutionProvider(str, Enum):
    """Supported ONNX Runtime execution providers.

    Only NVIDIA GPU-backed providers are supported:

    * ``CUDA`` — ``CUDAExecutionProvider``: standard GPU execution.
    * ``TENSORRT`` — ``TensorrtExecutionProvider`` with ``CUDAExecutionProvider``
      as fallback: enables TensorRT engine compilation for maximum GPU throughput.
    """

    CUDA = "cuda"
    TENSORRT = "tensorrt"


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
    """Backend that exports models to ONNX and runs inference with ONNX Runtime.

    Exports the model to a ``.onnx`` artifact at build time (trace or dynamo),
    then loads an ``onnxruntime.InferenceSession`` for inference.  Dynamic batch
    and spatial dimensions are inferred automatically from ``graph_spec``.

    Workflow::

        backend = ONNXRuntimeBackend()
        # build / tune as usual through ait.tune()
        ait.save(model, "model.ait")
        # later
        ait.load(model, "model.ait")
    """

    # State dictionary keys
    STATE_TYPE = "type"
    STATE_ONNX_MODEL_PATH = "onnx_model_path"
    STATE_ONNX_DATA_PATH = "onnx_data_path"
    STATE_DEVICE = "device"
    STATE_CONFIG = "config"
    STATE_OUTPUT_OBJECT = "output_object"
    STATE_GRAPH_SPEC = "graph_spec"
    STATE_SAMPLES = "samples"

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
        self._graph_spec: GraphSpec | None = None
        self._samples: SampleStore | None = None

    def key(self) -> str:
        """Returns the key of the backend."""
        return f"{self.__class__.__name__}_{self._config.key()}"

    def describe(self) -> str:
        """Returns the description of the backend."""
        return f"{self.__class__.__name__}({self._config.describe()})"

    def _build(self, module: nn.Module, graph_spec: GraphSpec, samples: SampleStore, cache_dir: Path) -> Backend:
        """Export the model to ONNX then load the session."""
        self._save_config(cache_dir)
        self._graph_spec = graph_spec

        self._output_object = self._get_output_object(module=module, sample=samples[0])

        module = module.eval().to(self._device)
        self._onnx_model_artifact = ArtifactPath(cache_dir, "model_raw.onnx")
        onnx_exporter = ONNXExporter(
            output_path=self._onnx_model_artifact.path,
            use_dynamo=self._config.use_dynamo,
            opset_version=self._config.opset_version,
        )
        onnx_exporter.export(module=module, sample=samples[0], graph_spec=graph_spec)

        data_file = Path(str(self._onnx_model_artifact.path) + ".data")
        if data_file.exists():
            self._onnx_data_artifact = ArtifactPath.from_existing(data_file, root=cache_dir)

        self._samples = samples
        offload(module, device="cpu")
        self._activate()

        return self

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

        Tensors are returned as-is (preserving their device); conversion to the
        format expected by ONNX Runtime happens in ``_infer`` via IOBinding.
        """
        session_input_names = {inp.name for inp in self._session.get_inputs()}
        inputs: dict[str, Any] = {}
        forward_inputs = self._graph_spec.forward_signature.normalize(args, kwargs)
        for locator, _ in self._graph_spec.input_spec.tensor_data:
            name = format_tensor_name(locator.path, "input")
            if name not in session_input_names:
                logger.debug("Input: %s not found in session inputs", name)
                continue
            inputs[name] = locator.get_value(forward_inputs.arguments)
        return inputs

    def _prepare_outputs(self, outputs: dict[str, torch.Tensor]) -> Any:
        """Reconstruct original output structure from session output tensors."""
        result = copy.deepcopy(self._output_object)
        for locator, _ in self._graph_spec.output_spec.tensor_data:
            name = format_tensor_name(locator.path, "output")
            if name in outputs:
                result = locator.set_value(result, outputs[name])
            else:
                logger.debug("Output: %s not found in session outputs", name)
        return result

    def _bind_inputs(self, io_binding: onnxruntime.IOBinding, inputs: dict[str, Any]) -> None:
        """Bind prepared inputs to an IOBinding handle.

        GPU tensors are bound zero-copy via DLPack; CPU tensors and other
        array-like values are bound as numpy arrays.
        """
        for name, value in inputs.items():
            value = value.contiguous()
            logger.debug("Binding input %s: device=%s shape=%s dtype=%s", name, value.device, value.shape, value.dtype)
            np_dtype = _TORCH_DTYPE_TO_NUMPY.get(value.dtype)
            if np_dtype is None:
                raise ValueError(f"Unsupported tensor dtype for ONNX Runtime IOBinding: {value.dtype}")
            io_binding.bind_input(
                name=name,
                device_type="cuda",
                device_id=value.device.index,
                element_type=np_dtype,
                shape=list(value.shape),
                buffer_ptr=value.data_ptr(),
            )

    def _bind_outputs(self, io_binding: onnxruntime.IOBinding) -> None:
        """Tell ORT to allocate all outputs on the CUDA device.

        ORT owns the output buffers; shapes are resolved at inference time.
        Tensors are retrieved after inference via ``_collect_outputs``.
        """
        device_id = self._device.index or 0
        for output in self._session.get_outputs():
            io_binding.bind_output(output.name, "cuda", device_id)

    def _collect_outputs(self, io_binding: onnxruntime.IOBinding) -> dict[str, torch.Tensor]:
        """Collect ORT CUDA outputs into torch tensors via D2D memcpy (no CPU round-trip)."""
        device = torch.device(self._device)
        return {
            node.name: memcpy_to_torch(ort_val.data_ptr(), list(ort_val.shape()), ort_val.data_type(), device)
            for node, ort_val in zip(self._session.get_outputs(), io_binding.get_outputs(), strict=False)
        }

    @nvtx.annotate(message="ONNXRuntimeBackend.infer", domain="AITune", color="green")
    def _infer(self, *args: Any, **kwargs: Any) -> Any:
        """Run inference through the ONNX Runtime session via IOBinding."""
        inputs = self._prepare_inputs(args, kwargs)
        io_binding = self._session.io_binding()
        self._bind_inputs(io_binding, inputs)
        self._bind_outputs(io_binding)
        self._session.run_with_iobinding(io_binding)
        return self._prepare_outputs(self._collect_outputs(io_binding))

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

    def _save_config(self, cache_dir: Path):
        """Store the backend configuration to a file."""
        config_path = cache_dir / "config.json"
        self._config.to_json(config_path)
        logger.info("Config saved to %s", config_path)

    def to_dict(self) -> dict:
        """Returns the state_dict of the backend."""
        if self._onnx_model_artifact is None:
            raise RuntimeError("Backend has not been built yet. Please call build() first.")
        state = {
            self.STATE_TYPE: self.__class__.__name__,
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
        backend._onnx_model_artifact = state_dict[cls.STATE_ONNX_MODEL_PATH]
        backend._onnx_data_artifact = state_dict.get(cls.STATE_ONNX_DATA_PATH)
        backend._device = state_dict[cls.STATE_DEVICE]
        backend._graph_spec = GraphSpec.from_dict(state_dict[cls.STATE_GRAPH_SPEC])
        backend._output_object = state_dict[cls.STATE_OUTPUT_OBJECT]
        samples_state = state_dict.get(cls.STATE_SAMPLES)
        backend._samples = SampleStore.from_dict(samples_state) if samples_state is not None else None
        backend.state = BackendState.CHECKPOINT_LOADED
        return backend
