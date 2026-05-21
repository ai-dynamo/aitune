# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Torch compile backend."""

import gc
from dataclasses import asdict, dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.nn as nn

from aitune.exceptions import AITuneError
from aitune.torch.backend.backend import Backend, BackendConfig, BackendState
from aitune.torch.libs.torch_compile import resolve_compile_dynamic
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.utils.cuda_utils import assert_is_available as assert_cuda_is_available

try:
    import torch_tensorrt
    from torch_tensorrt.dynamo import CompilationSettings as TorchTensorRTConfig
except (ImportError, RuntimeError, OSError):
    torch_tensorrt = None

    @dataclass
    class TorchTensorRTConfig:
        """Allows testing when torch_tensorrt is not installed."""

        enabled_precisions: set[torch.dtype] | None = None
        use_explicit_typing: bool = True
        workspace_size: int = 0


logger = getLogger(__name__)


def assert_torch_tensorrt():
    """Check if torch_tensorrt is installed."""
    if torch_tensorrt is None:
        raise ImportError("torch-tensorrt is not installed. Please install `torch-tensorrt`.")


@dataclass
class TorchTensorRTJitBackendConfig(BackendConfig):
    """Configuration for torch.compile(backend="torch_tensorrt")."""

    compile_config: TorchTensorRTConfig = field(  # pytype: disable=invalid-annotation
        default_factory=TorchTensorRTConfig
    )
    fullgraph: bool = False
    dynamic: bool | None = None
    autocast_enabled: bool = False
    autocast_dtype: torch.dtype | None = None

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
        """Convert TorchTensorRTJitBackendConfig to dict."""
        state_dict = asdict(self)
        state_dict["compile_config"] = asdict(self.compile_config)  # explicitly convert to dict
        return state_dict

    @classmethod
    def from_dict(cls, data: dict) -> "TorchTensorRTJitBackendConfig":
        """Initialise config from a plain dict (e.g. parsed from YAML).

        ``compile_config`` may be passed as a nested dict and will be
        reconstructed into a ``TorchTensorRTConfig`` instance automatically.
        """
        data = dict(data)
        if isinstance(data.get("compile_config"), dict):
            data["compile_config"] = TorchTensorRTConfig(**data["compile_config"])
        return cls(**data)


class TorchTensorRTJitBackend(Backend):
    """Torch TensorRT thought torch.compile(backend="torch_tensorrt").

    Backend does not use intermediate formats, and compiled model is not stored.
    """

    is_jit = True

    # State dictionary keys
    STATE_TYPE = "type"
    STATE_CONFIG = "config"
    STATE_ORIG_MODULE = "orig_module"
    STATE_DATA = "data"
    STATE_DEVICE = "device"
    STATE_COMPILE_DYNAMIC = "compile_dynamic"

    # Supported devices
    _devices: ClassVar[list[str]] = ["cuda"]

    def __init__(
        self,
        config: TorchTensorRTJitBackendConfig | None = None,
    ):
        """Initialize TorchTensorRTJitBackend.

        Args:
            config: Configuration for torch compile with torch_tensorrt
        """
        super().__init__()
        assert_cuda_is_available()
        assert_torch_tensorrt()

        # initialize variables
        self._config = config or TorchTensorRTJitBackendConfig()

        # build variables
        self._compiled_module = None
        self._orig_module = None
        self._data = None
        self._compile_dynamic = self._config.dynamic

    def key(self) -> str:
        """Returns the key of the backend."""
        return f"{self.__class__.__name__}_{self._config.key()}"

    def describe(self) -> str:
        """Returns the description of the backend."""
        return f"{self.__class__.__name__}({self._config.describe()})"

    def _build(self, module: nn.Module, graph_spec: GraphSpec, data: list[Sample], cache_dir: Path) -> Backend:
        """Build the model with Torch compile."""
        self._compile_dynamic = resolve_compile_dynamic(self._config.dynamic, graph_spec)
        self._save_config(cache_dir)
        module = module.eval()
        self._orig_module = module
        self._data = data
        self._compile()

        return self

    def _activate(self):
        """Activate backend."""
        self._compile()

    def _compile(self):
        """Compile module with Torch compile."""
        logger.info("Start compiling torch module.")
        torch._dynamo.reset()

        self._orig_module.to(self._device)

        self._compiled_module = torch.compile(
            self._orig_module,
            backend="torch_tensorrt",
            options=asdict(self._config.compile_config),
            fullgraph=self._config.fullgraph,
            dynamic=self._compile_dynamic,
        )

        for args, kwargs in self._data:
            self._compiled_module(*args, **kwargs)
        logger.info("Module has been compiled.")

    def _infer(self, *args: Any, **kwargs: Any) -> Any:
        """Run inference with TensorRT engine thought Torch compile.

        Args:
            *args: Inference arguments
            **kwargs: Inference keyword arguments

        Returns:
            Model outputs

        Raises:
            AITuneError: If backend is not activated
        """
        if not hasattr(self, "_compiled_module") or self._compiled_module is None:
            raise AITuneError("backend is not build. Please call build() first.")

        with (
            torch.autocast(
                device_type=str(self._device), dtype=self._config.autocast_dtype, enabled=self._config.autocast_enabled
            ),
            torch.no_grad(),
        ):
            return self._compiled_module(*args, **kwargs)

    def _deactivate(self):
        """Deactivate backend and cleanup."""
        self._compiled_module = None

    def _deploy(self):
        """Deploys the backend."""
        self._activate()
        self._data = None
        gc.collect()

    def _save_config(self, cache_dir: Path):
        """Store the backend configuration to a file."""
        config_path = cache_dir / "config.json"
        self._config.to_json(config_path)
        logger.info("Config saved to %s", config_path)

    def to_dict(self):
        """Returns the state_dict of the backend."""
        if self._orig_module is None:
            raise RuntimeError("Backend has not been properly initialized. Please call build() first.")
        return {
            self.STATE_TYPE: self.__class__.__name__,
            self.STATE_CONFIG: self._config.to_dict(),
            self.STATE_DATA: self._data,
            self.STATE_ORIG_MODULE: self._orig_module.state_dict(),
            self.STATE_DEVICE: self._device,
            self.STATE_COMPILE_DYNAMIC: self._compile_dynamic,
        }

    @classmethod
    def from_dict(cls, module: torch.nn.Module | None, state_dict: dict):
        """Creates a backend from a state_dict."""
        if state_dict.get(cls.STATE_TYPE) != cls.__name__:
            raise ValueError(f"Invalid state_dict type: {state_dict.get(cls.STATE_TYPE)}")

        if module is None:
            raise ValueError("Module is required to create a backend from a state_dict.")

        config = TorchTensorRTJitBackendConfig.from_dict(state_dict[cls.STATE_CONFIG])

        backend = cls(config=config)
        backend._data = state_dict[cls.STATE_DATA]
        backend._device = state_dict[cls.STATE_DEVICE]
        backend._compile_dynamic = state_dict.get(cls.STATE_COMPILE_DYNAMIC, config.dynamic)
        backend._orig_module = module
        module.load_state_dict(state_dict[cls.STATE_ORIG_MODULE], strict=False)
        backend.state = BackendState.CHECKPOINT_LOADED
        return backend
