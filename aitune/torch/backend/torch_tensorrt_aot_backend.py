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
"""TorchTensorRT backend with AOT compilation and intermediate model save."""

from dataclasses import asdict, dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar, Literal

import nvtx
import torch
import torch.nn as nn
from packaging.version import Version

from aitune.torch.backend.backend import Backend, BackendConfig, BackendState
from aitune.torch.config import DEFAULT_PICKLE_PROTOCOL
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.utils.cuda import assert_is_available as assert_cuda_is_available
from aitune.torch.utils.cuda import get_device as get_cuda_device

try:
    import torch_tensorrt
    from torch_tensorrt.dynamo import CompilationSettings as TorchTensorRTConfig
except (ImportError, RuntimeError, FileNotFoundError, OSError):
    torch_tensorrt = None

    @dataclass
    class TorchTensorRTConfig:
        """Allows testing when torch_tensorrt is not installed."""

        enabled_precisions: set[torch.dtype] = field(default_factory=lambda: {torch.float16})
        workspace_size: int = 0


logger = getLogger(__name__)

IRType = Literal["dynamo", "ts", "fx"]


def assert_torch_tensorrt():
    """Check if torch_tensorrt is installed."""
    if torch_tensorrt is None:
        raise ImportError("torch-tensorrt is not installed. Please install `torch-tensorrt`.")


def _is_primitive(value: Any) -> bool:
    """Check if a value is a primitive type (int, float, str, bool, None).

    Args:
        value: The value to check

    Returns:
        True if the value is a primitive type, False otherwise
    """
    return isinstance(value, int | float | str | bool)


@dataclass
class TorchTensorRTAotBackendConfig(BackendConfig):
    """Configuration for TorchTensorRTAotBackend.

    See torch_tensorrt/dynamo/_settings.py CompilationSettings for compile_config(TorchTensorRTConfig)
    """

    ir: IRType = "dynamo"
    compile_config: TorchTensorRTConfig = field(  # pytype: disable=invalid-annotation
        default_factory=lambda: TorchTensorRTConfig(enabled_precisions={torch.float16})
    )
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL

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
            device: Device to run the model on
        """
        super().__init__()
        assert_cuda_is_available()
        assert_torch_tensorrt()

        # initialize variables
        self._config = config or TorchTensorRTAotBackendConfig()

        # build variables
        self._opt_module = None
        self._exported_model_path = None

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
        data: list[Sample],
        cache_dir: Path,
    ) -> Backend:
        """Build the model with TensorRT."""
        logger.info("Start compiling torch module with TensorRT.")

        model = module.eval()
        model = model.to(self._device)

        self._save_config(cache_dir)

        # Set the device for the TensorRT backend compilation
        self._config.compile_config.device = torch_tensorrt.Device(get_cuda_device(self._device))

        # Note: torch export requires batch size to be greater then 1
        args, kwargs = data[0]

        if graph_spec.input_spec.has_batch_axis():
            max_batch_size = graph_spec.get_max_batch_size()
            batch_size = min(max_batch_size, 2)
            args, kwargs = graph_spec.input_spec.make_batch(args, kwargs, batch_size=batch_size)

        save_kwargs = {}
        if Version(torch.__version__) >= Version("2.7"):
            logger.info("Using pickle protocol %d.", self._config.pickle_protocol)
            save_kwargs["pickle_protocol"] = self._config.pickle_protocol

        # Create input signature for dynamic shapes
        logger.info("Preparing input signature for dynamic shapes.")
        input_signature = []
        for tensor_spec in graph_spec.input_spec.tensor_specs:
            input_i = torch_tensorrt.Input(
                min_shape=tensor_spec.min_shape,
                opt_shape=tensor_spec.max_shape,
                max_shape=tensor_spec.max_shape,
                dtype=tensor_spec.dtype,
                name=tensor_spec.name,
            )
            input_signature.append(input_i)

        # Extract non-tensor kwargs to pass as kwarg_inputs
        kwarg_inputs = {}
        for key, value in kwargs.items():
            if value is None:
                continue
            # Skip tensors - they should be in input_signature
            if isinstance(value, torch.Tensor):
                continue
            # Pass primitives and other types (lists, dicts, etc.) as kwarg_inputs
            if _is_primitive(value):
                kwarg_inputs[key] = [value]
            else:
                kwarg_inputs[key] = value

        logger.info("Compile model with Torch-TensorRT.")
        trt_model_compiled = torch_tensorrt.compile(
            model,
            ir=self._config.ir,
            inputs=input_signature,
            kwarg_inputs=kwarg_inputs,
            **asdict(self._config.compile_config),
        )

        self._exported_model_path = self._create_exported_model_path(cache_dir)
        torch_tensorrt.save(
            trt_model_compiled,
            self._exported_model_path.as_posix(),
            **save_kwargs,
        )

        logger.info("Module has been compiled and saved with TensorRT.")
        self._opt_module = trt_model_compiled
        return self

    def _activate(self):
        """Load compiled module."""
        self._opt_module = torch.export.load(self._exported_model_path.as_posix()).module().to(self._device)

    @nvtx.annotate(message="TorchTensorRTAotBackend.infer", domain="AITune", color="purple")
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

    def _create_exported_model_path(self, cache_dir: Path) -> Path:
        """Create path to exported model.

        In order to avoid name clashes we create a unique path for each model and graph spec.
        """
        path = cache_dir / "exported_model.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def to_dict(self) -> dict:
        """Returns the state_dict of the backend."""
        if self._exported_model_path is None:
            raise RuntimeError("No exported model path available. Model must be built first.")
        return {
            self.STATE_TYPE: self.__class__.__name__,
            self.EXPORTED_MODEL_PATH_KEY: self._exported_model_path,
            self.STATE_DEVICE: self._device,
        }

    @classmethod
    def from_dict(cls, module: torch.nn.Module | None, state_dict: dict) -> "TorchTensorRTAotBackend":
        """Creates a backend from a state_dict."""
        if state_dict.get(cls.STATE_TYPE) != cls.__name__:
            raise ValueError(f"Invalid state_dict type: {state_dict.get(cls.STATE_TYPE)}")

        backend = cls()  # create with default args, config is not needed
        backend._exported_model_path = state_dict[cls.EXPORTED_MODEL_PATH_KEY]
        backend._device = state_dict[cls.STATE_DEVICE]
        backend.state = BackendState.CHECKPOINT_LOADED
        return backend
