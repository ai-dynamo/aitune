# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Torch Inductor JIT backend."""

import gc
from copy import deepcopy
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from aitune.torch.backend.backend import Backend, BackendConfig, BackendState
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample

logger = getLogger(__name__)


@dataclass
class TorchInductorJitBackendConfig(BackendConfig):
    """Configuration for torch.compile with inductor backend.

    Args:
        fullgraph (bool): If False (default), torch.compile attempts to discover compileable regions
            in the function it will tune. If True, then we require the entire function to be
            captured into a single graph. If this is not possible (that is, if there are graph breaks),
            then this will raise an error.
        dynamic (bool or None): Use dynamic shape tracing. When this is True, we will up-front attempt
            to generate a kernel that is as dynamic as possible to avoid recompilations when
            sizes change. This may not always work as some operations/optimizations will
            force specialization; use TORCH_LOGS=dynamic to debug overspecialization.
            When this is False, we will NEVER generate dynamic kernels, we will always specialize.
            By default (None), we automatically detect if dynamism has occurred and compile a more
            dynamic kernel upon recompile.
        mode (str): Can be either "default", "reduce-overhead", "max-autotune" or "max-autotune-no-cudagraphs"

            - "default" is the default mode, which is a good balance between performance and overhead

            - "reduce-overhead" is a mode that reduces the overhead of python with CUDA graphs,
              useful for small batches. Reduction of overhead can come at the cost of more memory
              usage, as we will cache the workspace memory required for the invocation so that we
              do not have to reallocate it on subsequent runs. Reduction of overhead is not guaranteed
              to work; today, we only reduce overhead for CUDA only graphs which do not mutate inputs.
              There are other circumstances where CUDA graphs are not applicable; use TORCH_LOG=perf_hints
              to debug.

            - "max-autotune" is a mode that leverages Triton or template based matrix multiplications
              on supported devices and Triton based convolutions on GPU.
              It enables CUDA graphs by default on GPU.

            - "max-autotune-no-cudagraphs" is a mode similar to "max-autotune" but without CUDA graphs

            - To see the exact configs that each mode sets you can call `torch._inductor.list_mode_options()`

        options (dict): A dictionary of options to pass to the backend.
            - To see the full list of configs that it supports by calling `torch._inductor.list_options()`

        autocast_enabled (bool): If True, enable autocast.

        autocast_dtype (torch.dtype): The dtype to use for autocast.
    """

    fullgraph: bool = False
    dynamic: bool | None = None
    mode: str | None = None
    options: dict[str, str | int | bool] | None = None
    autocast_enabled: bool = False
    autocast_dtype: torch.dtype | None = None

    def __post_init__(self):
        """Post init."""
        # Check that mode and options are not both supplied in config
        if self.mode is not None and self.options is not None:
            raise ValueError(
                "Cannot specify both 'mode' and 'options' parameters in config. "
                "Use either 'mode' for predefined configurations or 'options' "
                "for custom configurations, but not both."
            )


class TorchInductorJitBackend(Backend):
    """Backend that does torch compilation with Inductor."""

    is_jit = True

    # State dictionary keys
    STATE_TYPE = "type"
    STATE_CONFIG = "config"
    STATE_ORIG_MODULE = "orig_module"
    STATE_DATA = "data"
    STATE_OUTPUT_DTYPE = "output_dtype"
    STATE_DEVICE = "device"

    def __init__(
        self,
        config: TorchInductorJitBackendConfig | None = None,
    ):
        """Initializes backend.

        Args:
            config: Configuration for torch compile with inductor backend
        """
        super().__init__()

        # initialize variables
        self._config = config or TorchInductorJitBackendConfig()

        # build variables
        self._compiled_module = None
        self._orig_module = None
        self._output_dtype = None
        self._data = None

    def key(self) -> str:
        """Returns the key of the backend."""
        return f"{self.__class__.__name__}_{self._config.key()}"

    def describe(self) -> str:
        """Returns the description of the backend."""
        return f"{self.__class__.__name__}({self._config.describe()})"

    @staticmethod
    def _get_dtype(module: nn.Module, data: list[Sample]) -> torch.dtype:
        """Get the output dtype of the module by running a sample inference.

        Args:
            module (nn.Module): The module to get the dtype from.
            data (list[Sample]): List of sample inputs to run through the module.

        Returns:
            torch.dtype: The dtype of the module's output tensor. Returns None if output is not a tensor.
        """
        args, kwargs = data[0]
        res = module(*deepcopy(args), **deepcopy(kwargs))
        if isinstance(res, torch.Tensor):
            return res.dtype
        else:
            return None

    def _build(self, module: nn.Module, graph_spec: GraphSpec, data: list[Sample], cache_dir: Path) -> Backend:
        """Builds the model with torch.compile."""
        self._save_config(cache_dir)

        module.to(self._device)
        self._output_dtype = self._get_dtype(module, data)
        self._orig_module = module
        self._data = data
        self._compile()
        return self

    def _compile(self):
        logger.info("Start compiling torch module.")
        with (
            torch.autocast(
                device_type=str(self._device.type),
                dtype=self._config.autocast_dtype,
                enabled=self._config.autocast_enabled,
            ),
            torch.no_grad(),
        ):
            self._orig_module.to(self._device)
            self._compiled_module = torch.compile(
                self._orig_module,
                fullgraph=self._config.fullgraph,
                dynamic=self._config.dynamic,
                mode=self._config.mode,
                options=self._config.options,
            )

            for args, kwargs in self._data:
                self._compiled_module(*deepcopy(args), **deepcopy(kwargs))
        logger.info("Module has been compiled.")

    def _activate(self):
        """Activates backend."""
        if self._compiled_module is None:  # TBD pb: after introducing module states this should be changed
            self._compile()

    def _infer(self, *args: Any, **kwargs: Any) -> Any:
        """Runs inference with the given arguments.

        Args:
            *args: inference arguments
            **kwargs: inference keyword arguments

        Returns:
            Any: The result of the inference.
        """
        with (
            torch.autocast(
                device_type=str(self._device),
                dtype=self._config.autocast_dtype,
                enabled=self._config.autocast_enabled,
            ),
            torch.no_grad(),
        ):
            res = self._compiled_module(*args, **kwargs)
            if isinstance(res, torch.Tensor) and res.dtype != self._output_dtype and self._output_dtype is not None:
                # autocast changed the dtype of the output, converting back to the original dtype
                return res.to(self._output_dtype)
            else:
                return res

    def _deactivate(self):
        """Deactivates backend."""
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
        if not self._orig_module:
            raise RuntimeError("Backend has not been properly initialized. Please call build() first.")

        return {
            self.STATE_TYPE: self.__class__.__name__,
            self.STATE_CONFIG: self._config.to_dict(),
            self.STATE_OUTPUT_DTYPE: self._output_dtype,
            self.STATE_DATA: self._data,
            self.STATE_ORIG_MODULE: self._orig_module.state_dict(),
            self.STATE_DEVICE: self._device,
        }

    @classmethod
    def from_dict(cls, module: torch.nn.Module | None, state_dict: dict):
        """Creates a backend from a state_dict."""
        if state_dict.get(cls.STATE_TYPE) != cls.__name__:
            raise ValueError(f"Invalid state_dict type: {state_dict.get(cls.STATE_TYPE)}")

        if module is None:
            raise ValueError("Module is required to create a backend from a state_dict.")

        config = TorchInductorJitBackendConfig.from_dict(state_dict[cls.STATE_CONFIG])

        backend = cls(config=config)
        backend._output_dtype = state_dict[cls.STATE_OUTPUT_DTYPE]
        backend._data = state_dict[cls.STATE_DATA]
        backend._device = state_dict[cls.STATE_DEVICE]
        backend._orig_module = module
        module.load_state_dict(state_dict[cls.STATE_ORIG_MODULE], strict=False)
        backend.state = BackendState.CHECKPOINT_LOADED
        return backend
