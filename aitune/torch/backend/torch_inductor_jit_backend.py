# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Torch Inductor JIT backend."""

import gc
from collections.abc import Sequence
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, get_args

import torch
import torch.nn as nn

from aitune.torch.backend.backend import Backend, BackendConfig, BackendState, BuildMode, ExecutionMode
from aitune.torch.libs.torch_compile import TorchCompileMode, resolve_compile_dynamic
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import Sample, SampleStore
from aitune.torch.utils.module import move_module_to_device

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
        mode (TorchCompileMode or None): Can be either "default", "reduce-overhead", "max-autotune"
            or "max-autotune-no-cudagraphs".

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

        Note: inference is done with torch.no_grad() context. The torch.inference_mode() context must not be used
        as it would require outputs from a model to be used with same inference mode - this would be confusing to a user
        and required code changes from the user.

    """

    fullgraph: bool = False
    dynamic: bool | None = None
    mode: TorchCompileMode | None = None
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
        if self.mode is not None and self.mode not in get_args(TorchCompileMode):
            raise ValueError(f"Invalid mode: {self.mode!r}. Supported values: {get_args(TorchCompileMode)}")


class TorchInductorJitBackend(Backend):
    """Backend that does torch compilation with Inductor."""

    _build_mode = BuildMode.JUST_IN_TIME
    _execution_modes = frozenset({ExecutionMode.SINGLE_GPU, ExecutionMode.MULTI_GPU})

    # State dictionary keys
    STATE_TYPE = "type"
    STATE_CONFIG = "config"
    STATE_ORIG_MODULE = "orig_module"
    STATE_DATA = "data"
    STATE_SAMPLES = "samples"
    STATE_OUTPUT_DTYPE = "output_dtype"
    STATE_DEVICE = "device"
    STATE_COMPILE_DYNAMIC = "compile_dynamic"

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
        self._required_casting_dtype = None
        self._samples: SampleStore | None = None
        # Kept only for loading checkpoints written before samples became disk-backed.
        self._data = None
        self._compile_dynamic = self._config.dynamic
        self._infer_impl = self._infer_no_autocast

    def key(self) -> str:
        """Returns the key of the backend."""
        return f"{self.__class__.__name__}_{self._config.key()}"

    def describe(self) -> str:
        """Returns the description of the backend."""
        return f"{self.__class__.__name__}({self._config.describe()})"

    def _get_required_casting_dtype(self, module: nn.Module, samples: Sequence[Sample]) -> torch.dtype | None:
        """Get the required casting dtype of the module by running a sample inference with and without autocast.

        If the dtype of the output is different with and without autocast, return the dtype of the output without autocast.
        Otherwise, return None.

        Args:
            module (nn.Module): The module to get the dtype from.
            samples (Sequence[Sample]): Sample inputs to run through the module.

        Returns:
            torch.dtype: The required casting dtype. Returns None if no casting is required.
        """
        with torch.no_grad():
            with torch.autocast(
                device_type=str(self._device.type),
                dtype=self._config.autocast_dtype,
                enabled=True,
            ):
                args, kwargs = samples[0]
                autocast_result = module(*args, **kwargs)

            if isinstance(autocast_result, torch.Tensor):
                args, kwargs = samples[0]
                orig_result = module(*args, **kwargs)
                if orig_result.dtype != autocast_result.dtype:
                    return orig_result.dtype

        return None

    def _build(self, module: nn.Module, graph_spec: GraphSpec, samples: SampleStore, cache_dir: Path) -> Backend:
        """Builds the model with torch.compile."""
        self._compile_dynamic = resolve_compile_dynamic(self._config.dynamic, graph_spec)
        self._save_config(cache_dir)

        move_module_to_device(module, self._device)
        self._orig_module = module
        if self._config.autocast_enabled:
            self._required_casting_dtype = self._get_required_casting_dtype(module, samples)
        self._samples = samples
        self._compile()
        self._activate()
        return self

    def _compile(self):
        logger.info("Start compiling torch module.")
        move_module_to_device(self._orig_module, self._device)

        self._compiled_module = torch.compile(
            self._orig_module,
            fullgraph=self._config.fullgraph,
            dynamic=self._compile_dynamic,
            mode=self._config.mode,
            options=self._config.options,
        )

        self._select_infer_impl()
        for args, kwargs in self._iter_samples():
            self._infer_impl(*args, **kwargs)
        logger.info("Module has been compiled.")

    def _activate(self):
        """Activates backend."""
        if self._compiled_module is None:  # TBD pb: after introducing module states this should be changed
            self._compile()
        self._select_infer_impl()

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
                device_type=str(self._device.type),
                dtype=self._config.autocast_dtype,
                enabled=True,
            ):
                res = self._compiled_module(*args, **kwargs)
                if self._required_casting_dtype is not None:
                    # autocast changed the dtype of the output, converting back to the original dtype
                    return res.to(self._required_casting_dtype)
        return res

    def _infer_no_autocast(self, *args: Any, **kwargs: Any) -> Any:
        """Runs inference with the given arguments. Does not use autocast.

        Args:
            *args: inference arguments
            **kwargs: inference keyword arguments

        Returns:
            Any: The result of the inference.
        """
        with torch.no_grad():
            return self._compiled_module(*args, **kwargs)

    def _select_infer_impl(self):
        """Select the inference implementation based on configuration."""
        self._infer_impl = self._infer_with_autocast if self._config.autocast_enabled else self._infer_no_autocast

    def _infer(self, *args: Any, **kwargs: Any) -> Any:
        """Run inference using the implementation selected for the configuration.

        Args:
            *args: inference arguments
            **kwargs: inference keyword arguments

        Returns:
            Any: The result of the inference.
        """
        return self._infer_impl(*args, **kwargs)

    def _deactivate(self):
        """Deactivates backend."""
        self._compiled_module = None

    def _deploy(self):
        """Deploys the backend."""
        self._activate()
        self._samples = None
        self._data = None
        gc.collect()

    def _iter_samples(self):
        """Iterate persisted samples, with support for legacy inline checkpoint data."""
        if self._samples is not None:
            return self._samples.iter_samples(self._device)
        if self._data is not None:
            return iter(self._data)
        raise RuntimeError("Backend has no warmup data. Please call build() first.")

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
            self.STATE_OUTPUT_DTYPE: self._required_casting_dtype,
            self.STATE_SAMPLES: self._samples.to_dict() if self._samples is not None else None,
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

        config = TorchInductorJitBackendConfig.from_dict(state_dict[cls.STATE_CONFIG])

        backend = cls(config=config)
        backend._required_casting_dtype = state_dict[cls.STATE_OUTPUT_DTYPE]
        samples_state = state_dict.get(cls.STATE_SAMPLES)
        backend._samples = SampleStore.from_dict(samples_state) if samples_state is not None else None
        backend._data = state_dict.get(cls.STATE_DATA)
        backend._device = state_dict[cls.STATE_DEVICE]
        backend._compile_dynamic = state_dict.get(cls.STATE_COMPILE_DYNAMIC, config.dynamic)
        backend._orig_module = module
        module.load_state_dict(state_dict[cls.STATE_ORIG_MODULE], strict=False)
        backend.state = BackendState.CHECKPOINT_LOADED
        return backend
