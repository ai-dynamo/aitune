# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch eager backend."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from aitune.torch.backend.backend import Backend, BackendConfig, BackendState
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample


@dataclass
class TorchEagerBackendConfig(BackendConfig):
    """Configuration for torch eager backend.

    Args:
        autocast_enabled (bool): If True, enable autocast.
        autocast_dtype (torch.dtype): The dtype to use for autocast.
    """

    autocast_enabled: bool = False
    autocast_dtype: torch.dtype | None = None


class TorchEagerBackend(Backend):
    """Backend that runs the model in eager mode with/without autocast.

    Note: inference is done with torch.no_grad() context. The torch.inference_mode() context must not be used
    as it would require outputs from a model to be used with same inference mode - this would be confusing to a user
    and required code changes from the user.
    """

    # State dictionary keys
    STATE_TYPE = "type"
    STATE_CONFIG = "config"
    STATE_ORIG_MODULE = "orig_module"
    STATE_DEVICE = "device"
    STATE_OUTPUT_DTYPE = "output_dtype"

    def __init__(self, config: TorchEagerBackendConfig | None = None):
        """Initializes backend."""
        super().__init__()
        self._config = config or TorchEagerBackendConfig()
        self._orig_module = None
        self._graph_spec = None
        self._required_casting_dtype = None

    def is_jit(self) -> bool:
        """Returns True if the backend is a JIT backend."""
        return True

    def key(self) -> str:
        """Returns the key of the backend."""
        return f"{self.__class__.__name__}_{self._config.key()}"

    def describe(self) -> str:
        """Returns the description of the backend."""
        return f"{self.__class__.__name__}({self._config.describe()})"

    def _get_required_casting_dtype(self, module: nn.Module, data: list[Sample]) -> torch.dtype | None:
        """Get the required casting dtype of the module by running a sample inference with and without autocast.

        If the dtype of the output is different with and without autocast, return the dtype of the output without autocast.
        Otherwise, return None.

        Args:
            module (nn.Module): The module to get the dtype from.
            data (list[Sample]): List of sample inputs to run through the module.

        Returns:
            torch.dtype: The required casting dtype. Returns None if no casting is required.
        """
        with torch.no_grad():
            with torch.autocast(
                device_type=str(self._device),
                dtype=self._config.autocast_dtype,
                enabled=True,
            ):
                args, kwargs = deepcopy(data[0])
                autocast_result = self._orig_module(*args, **kwargs)

            if isinstance(autocast_result, torch.Tensor):
                args, kwargs = deepcopy(data[0])
                orig_result = self._orig_module(*args, **kwargs)
                if orig_result.dtype != autocast_result.dtype:
                    return orig_result.dtype

        return None

    def _build(self, module: nn.Module, graph_spec: GraphSpec, data: list[Sample], cache_dir: Path) -> Backend:
        """Builds the model."""
        self._save_config(cache_dir)
        module.to(self._device)

        self._orig_module = module
        self._graph_spec = graph_spec
        if self._config.autocast_enabled:
            self._required_casting_dtype = self._get_required_casting_dtype(module, data)

        # must be activated before returning the backend
        self._activate()

        return self

    def _activate(self):
        """Activates runner."""
        if self._config.autocast_enabled:
            self._infer = self._infer_with_autocast

    def _infer_with_autocast(self, *args: Any, **kwargs: Any) -> Any:
        """Runs inference with the given arguments.

        Args:
            *args: inference arguments
            **kwargs: inference keyword arguments

        Returns:
            Any: The result of the inference.
        """
        with torch.no_grad():
            with torch.autocast(
                device_type=str(self._device),
                dtype=self._config.autocast_dtype,
                enabled=True,
            ):
                res = self._orig_module(*args, **kwargs)
                if self._required_casting_dtype is not None:
                    # autocast changed the dtype of the output, converting back to the original dtype
                    return res.to(self._required_casting_dtype)
        return res

    def _infer(self, *args: Any, **kwargs: Any) -> Any:
        """Runs inference with the given arguments. Does not use autocast.

        It can be replaced at runtime by _infer_with_autocast.

        Args:
            *args: inference arguments
            **kwargs: inference keyword arguments

        Returns:
            Any: The result of the inference.
        """
        with torch.no_grad():
            return self._orig_module(*args, **kwargs)

    def _deactivate(self):
        """Deactivates runner."""
        pass

    def _deploy(self):
        """Deploys the backend."""
        self._orig_module.to(self._device)
        if self._config.autocast_enabled:
            self._infer = self._infer_with_autocast

    def _save_config(self, cache_dir: Path):
        """Store the backend configuration to a file."""
        config_path = cache_dir / "config.json"
        self._config.to_json(config_path)

    def to_dict(self):
        """Returns the state_dict of the backend."""
        if self._orig_module is None:
            raise RuntimeError("Backend has not been properly initialized. Please call build() first.")
        return {
            self.STATE_TYPE: self.__class__.__name__,
            self.STATE_CONFIG: self._config.to_dict(),
            self.STATE_ORIG_MODULE: self._orig_module.state_dict(),
            self.STATE_OUTPUT_DTYPE: self._required_casting_dtype,
            self.STATE_DEVICE: self.device,
        }

    @classmethod
    def from_dict(cls, module: torch.nn.Module | None, state_dict: dict):
        """Creates a backend from a state_dict."""
        if state_dict.get(cls.STATE_TYPE) != cls.__name__:
            raise ValueError(f"Invalid state_dict type: {state_dict.get(cls.STATE_TYPE)}")

        if module is None:
            raise ValueError("Module is required to create a backend from a state_dict.")

        config = TorchEagerBackendConfig.from_dict(state_dict[cls.STATE_CONFIG])

        backend = cls(config=config)
        backend._required_casting_dtype = state_dict[cls.STATE_OUTPUT_DTYPE]
        backend._set_device(state_dict[cls.STATE_DEVICE])
        backend._orig_module = module
        module.load_state_dict(state_dict[cls.STATE_ORIG_MODULE], strict=False)
        backend.state = BackendState.CHECKPOINT_LOADED
        return backend
