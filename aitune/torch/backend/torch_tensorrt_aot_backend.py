# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TorchTensorRT backend with AOT compilation and intermediate model save."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar, cast

import torch
import torch.nn as nn

from aitune.torch.backend.backend import Backend, BackendBuildStep, BackendConfig, BackendState
from aitune.torch.checkpoint.artifact import ArtifactPath
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import Sample
from aitune.torch.utils.cuda_utils import assert_is_available as assert_cuda_is_available
from aitune.torch.utils.cuda_utils import get_device as get_cuda_device
from aitune.torch.utils.shapes import build_dynamic_shapes, prepare_export_sample, print_dynamic_shapes
from aitune.torch.utils.tensor import format_tensor_name

try:
    import torch_tensorrt
    from torch_tensorrt.dynamo import CompilationSettings as TorchTensorRTConfig
except (ImportError, RuntimeError, OSError):
    torch_tensorrt = None

    @dataclass
    class TorchTensorRTConfig:
        """Fallback settings matching upstream defaults when torch_tensorrt is unavailable.

        Attributes:
            enabled_precisions: Enabled TensorRT precisions. Defaults to None.
            use_explicit_typing: Whether to use explicit typing. Defaults to True.
            workspace_size: TensorRT workspace size. Defaults to 0.
        """

        enabled_precisions: set[torch.dtype] | None = None
        use_explicit_typing: bool = True
        workspace_size: int = 0


logger = getLogger(__name__)


class TorchTensorRTAotBuildStep(BackendBuildStep):
    """Identifiers for discrete sub-steps of a TorchTensorRTAot backend build."""

    TORCH_EXPORT = "Torch export"
    TORCHTRT_COMPILE = "Torch-TensorRT compile"
    COMPILED_MODEL_SAVE = "Compiled model save"


def assert_torch_tensorrt():
    """Check if torch_tensorrt is installed."""
    if torch_tensorrt is None:
        raise ImportError("torch-tensorrt is not installed. Please install `torch-tensorrt`.")


def _is_primitive(value: Any) -> bool:
    """Check if a value is a primitive type (int, float, str, bool)."""
    return isinstance(value, int | float | str | bool)


@dataclass
class TorchTensorRTAotBackendConfig(BackendConfig):
    """Configuration for TorchTensorRTAotBackend.

    See torch_tensorrt/dynamo/_settings.py CompilationSettings for compile_config(TorchTensorRTConfig)

    Attributes:
        compile_config: Torch-TensorRT compilation settings. Defaults to TorchTensorRTConfig().
        pickle_protocol: Pickle protocol for saved compiled artifacts. Defaults to 5.
    """

    compile_config: TorchTensorRTConfig = field(  # pytype: disable=invalid-annotation
        default_factory=TorchTensorRTConfig
    )
    pickle_protocol: int = 5

    def describe(self) -> str:
        """Describe the backend configuration. Display only changed fields."""
        other = self.__class__()
        compile_config_parts = self._get_changed_fields(
            self.compile_config, other.compile_config, include=["enabled_precisions"]
        )
        parts = [f"compile_config=TorchTensorRTConfig({','.join(compile_config_parts)})"]

        changed_fields = self._get_changed_fields(self, other, exclude=["compile_config"])
        parts.extend(changed_fields)

        return ",".join(parts)

    def to_dict(self):
        """Saves the backend configuration to a file."""
        state_dict = asdict(self)
        state_dict["compile_config"] = asdict(self.compile_config)  # explicitly convert to dict
        return state_dict

    @classmethod
    def from_dict(cls, data: dict) -> "TorchTensorRTAotBackendConfig":
        """Initialise config from a plain dict (e.g. parsed from YAML).

        ``compile_config`` may be passed as a nested dict and will be
        reconstructed into a ``TorchTensorRTConfig`` instance automatically.
        """
        data = dict(data)
        data.pop("ir", None)
        if isinstance(data.get("compile_config"), dict):
            data["compile_config"] = TorchTensorRTConfig(**data["compile_config"])
        return cls(**data)


class TorchTensorRTAotBackend(Backend):
    """Backend that compiles model using TensorRT."""

    is_jit = False

    # State dictionary keys
    STATE_TYPE = "type"
    EXPORTED_MODEL_PATH_KEY = "exported_model_path"
    STATE_DEVICE = "device"

    # Supported devices
    _devices: ClassVar[list[str]] = ["cuda"]

    def __init__(
        self,
        config: TorchTensorRTAotBackendConfig | None = None,
    ):
        """Initialize TensorRT backend.

        Args:
            config: Configuration for TensorRT compilation
        """
        super().__init__()
        assert_cuda_is_available()
        assert_torch_tensorrt()

        # initialize variables
        self._config = config or TorchTensorRTAotBackendConfig()

        # build variables
        self._opt_module = None
        self._exported_model_artifact: ArtifactPath | None = None

    def key(self) -> str:
        """Returns the key of the backend."""
        return f"{self.__class__.__name__}_{self._config.key()}"

    def describe(self) -> str:
        """Returns the description of the backend."""
        return f"{self.__class__.__name__}({self._config.describe()})"

    def _build(
        self,
        module: nn.Module,
        graph_spec: GraphSpec,
        data: Sequence[Sample],
        cache_dir: Path,
    ) -> Backend:
        """Build the model with TensorRT.

        Exports the module via ``torch.export.export`` with shared ``Dim`` instances
        across inputs, then hands the resulting ``ExportedProgram`` to
        ``torch_tensorrt.dynamo.compile``. This replaces the previous flow that called
        ``torch_tensorrt.compile`` and let it run its own export, which built one
        independent ``Dim`` per ``Input`` and raised ``ConstraintViolationError``
        whenever two inputs had to share a runtime axis (e.g. HF encoders' shared
        batch on ``input_ids`` and ``attention_mask``).

        Workaround pending a ``dynamic_shapes`` passthrough on ``torch_tensorrt.compile``;
        revert to ``torch_tensorrt.compile(...)`` once that lands upstream.
        """
        logger.info("Start compiling torch module with TensorRT.")

        model = module.eval()
        model = model.to(self._device)

        self._save_config(cache_dir)

        # Set the device for the TensorRT backend compilation
        self._config.compile_config.device = torch_tensorrt.Device(get_cuda_device(self._device))

        args, kwargs = prepare_export_sample(data[0], graph_spec)
        args = tuple(a.to(self._device) if isinstance(a, torch.Tensor) else a for a in args)
        kwargs = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

        dynamic_shapes = build_dynamic_shapes((args, kwargs), graph_spec, use_auto=False)
        # All entries empty/None → fully static graph; pass None to torch.export.export
        # so it short-circuits the dynamic-shape resolution path.
        if not any(dynamic_shapes.values()):
            dynamic_shapes = None

        if dynamic_shapes is not None:
            print_dynamic_shapes(dynamic_shapes)

        with self._track_build_step(TorchTensorRTAotBuildStep.TORCH_EXPORT):
            logger.info("Exporting model with torch.export.export.")
            with torch.no_grad():
                exported = torch.export.export(
                    model,
                    args,
                    kwargs=kwargs if kwargs else None,
                    dynamic_shapes=dynamic_shapes,
                    strict=False,
                )

        # Optimization-profile inputs for TRT engine building.
        input_signature = []
        for locator, tensor_spec in graph_spec.input_spec.tensor_data:
            input_name = format_tensor_name(locator.path, "input")
            min_shape, opt_shape, max_shape = graph_spec.get_effective_input_shapes(locator, tensor_spec)
            input_i = torch_tensorrt.Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=tensor_spec.dtype,
                name=input_name,
            )
            input_signature.append(input_i)
            logger.info(
                "Torch-TensorRT input profile %s: min=%s opt=%s max=%s dtype=%s",
                input_name,
                min_shape,
                opt_shape,
                max_shape,
                tensor_spec.dtype,
            )

        # Pass non-tensor kwargs through; tensors are already captured by the ExportedProgram.
        kwarg_inputs = {}
        for key, value in kwargs.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                continue
            if _is_primitive(value):
                kwarg_inputs[key] = [value]
            else:
                kwarg_inputs[key] = value

        with self._track_build_step(TorchTensorRTAotBuildStep.TORCHTRT_COMPILE):
            logger.info("Compile model with Torch-TensorRT.")
            trt_model_compiled = torch_tensorrt.dynamo.compile(
                exported,
                inputs=input_signature,
                kwarg_inputs=kwarg_inputs,
                **asdict(self._config.compile_config),
            )

        with self._track_build_step(TorchTensorRTAotBuildStep.COMPILED_MODEL_SAVE) as result:
            self._exported_model_artifact = self._create_exported_model_artifact(cache_dir)
            torch_tensorrt.save(
                trt_model_compiled,
                self._exported_model_artifact.path.as_posix(),
                retrace=False,
                pickle_protocol=self._config.pickle_protocol,
            )
            result["compiled_model_size_bytes"] = self._exported_model_artifact.path.stat().st_size

        logger.info("Module has been compiled and saved with TensorRT.")
        self._opt_module = trt_model_compiled
        return self

    def _activate(self):
        """Load compiled module."""
        exported_model_artifact = cast(ArtifactPath, self._exported_model_artifact)
        self._opt_module = torch_tensorrt.load(exported_model_artifact.path.as_posix()).module().to(self._device)

    def _infer(self, *args: Any, **kwargs: Any) -> Any:
        """Run inference with TensorRT engine.

        Args:
            *args: Inference arguments
            **kwargs: Inference keyword arguments

        Returns:
            Model outputs
        """
        if self._opt_module is None:
            raise RuntimeError("Module is not activated. Please call activate() first.")

        with torch.no_grad():
            return self._opt_module(*args, **kwargs)

    def _deactivate(self):
        """Deactivate backend and cleanup."""
        self._opt_module = None

    def _deploy(self):
        """Deploys the backend."""
        self._activate()

    def _save_config(self, cache_dir: Path):
        """Store the backend configuration to a file."""
        config_path = cache_dir / "config.json"
        self._config.to_json(config_path)
        logger.info("Config saved to %s", config_path)

    def _create_exported_model_artifact(self, cache_dir: Path) -> ArtifactPath:
        """Create the exported model artifact.

        In order to avoid name clashes we create a unique path for each model and graph spec.
        """
        cache_dir.mkdir(parents=True, exist_ok=True)
        return ArtifactPath(cache_dir, "exported_model.pt")

    def to_dict(self) -> dict:
        """Returns the state_dict of the backend."""
        if self._exported_model_artifact is None:
            raise RuntimeError("No exported model path available. Model must be built first.")
        return {
            self.STATE_TYPE: self.__class__.__name__,
            self.EXPORTED_MODEL_PATH_KEY: self._exported_model_artifact,
            self.STATE_DEVICE: self._device,
        }

    @classmethod
    def from_dict(cls, module: torch.nn.Module | None, state_dict: dict) -> "TorchTensorRTAotBackend":
        """Creates a backend from a state_dict."""
        if state_dict.get(cls.STATE_TYPE) != cls.__name__:
            raise ValueError(f"Invalid state_dict type: {state_dict.get(cls.STATE_TYPE)}")

        backend = cls()  # create with default args, config is not needed
        backend._exported_model_artifact = state_dict[cls.EXPORTED_MODEL_PATH_KEY]
        backend._device = state_dict[cls.STATE_DEVICE]
        backend.state = BackendState.CHECKPOINT_LOADED
        return backend
