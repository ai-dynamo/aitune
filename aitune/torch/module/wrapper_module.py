# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AITune wrapper module."""

import gc
from collections import OrderedDict
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Any, cast

import torch
import wrapt

from aitune.torch.backend.backend import Backend
from aitune.torch.config import aitune_cache_dir
from aitune.torch.config import config as global_config
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.passthrough_module import PassthroughModule
from aitune.torch.module.recording_module import RecordingModule
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.module.tuned_module import TunedModule
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.tune_strategy import FirstWinsStrategy
from aitune.torch.tune_strategy.tune_strategy import (
    DummyTuneStrategy,
    TuneStrategy,
)
from aitune.torch.utils.module import get_module_device, offload
from aitune.utils.monitoring import annotate

logger = getLogger(__name__)

DEFAULT_STRATEGY = FirstWinsStrategy()


class ModuleState(Enum):
    """Possible states of the Module class."""

    INIT = "init"
    PASSTHROUGH = "passthrough"
    RECORDING = "recording"
    TUNED = "tuned"


StrategyList = list[TuneStrategy]
StrategyMap = dict[SampleMetadata, TuneStrategy]


class Module(wrapt.CallableObjectProxy):
    """AITune module wrapper.

    This class wraps a torch module and provides tuning functionality.
    The module can be in 3 different states:
    - `passthrough`: the module is not tuned, and will behave identically to the original module.
    - `recording`: the module records samples and detects multi-graphs, this is a necessary step before tuning.
    - `tuned`: the module is tuned and uses underlying tuned module.

    You can go from passthrough and recording with `enablePassthrough` and `enableRecording` methods.

    You can go from tuned to passthrough/recording only if the force flag is True - then the module will be reset.

    This wrapper can be used in place of a torch module, and will behave identically to the original module.

    Example:
        >>> model = torch.nn.Linear(10, 10)
        >>> model = Module(model, "name")
    """

    TYPE_KEY = "type"
    NAME_KEY = "name"
    TUNED_MODULE_KEY = "tuned_module"

    def __init__(
        self,
        module: torch.nn.Module,
        name: str | None = None,
        strategy: TuneStrategy | None = None,
        strategies: StrategyList | StrategyMap | None = None,
    ):
        """Initializes module.

        Args:
            module: torch module to wrap.
            name: module name which differences from other tuned modules.
            strategy: optional strategy, which will be used for all encountered graphs
            strategies: list or dict of strategies, each for a graph
        """
        super().__init__(module)
        self._self_name = sanitize_model_name(name) or get_object_name(module)
        self._self_orig_forward = module.forward
        self._original_forward_pre_hooks = module._forward_pre_hooks
        self._original_forward_hooks = module._forward_hooks
        self._self_proxy_forward = wrapt.decorator(self._forward)(module.forward)
        module.forward = self._self_proxy_forward
        self._setup_strategies(strategy, strategies)

        self._self_device = get_module_device(module) or torch.device("cpu")
        self._self_state = ModuleState.INIT
        self._self_wrapper = None
        self._self_prev_recording = None

        MODULE_REGISTRY.register(self._self_name, self)

    def __getitem__(self, key: Any) -> Any:
        """Delegate __getitem__ calls to the wrapped module.

        This allows the proxy to handle indexing operations on the wrapped module,
        which is particularly useful for Sequential modules and other indexable modules.

        Args:
            key: The index or key to use for accessing the wrapped module.

        Returns:
            The result of accessing the wrapped module with the given key.
        """
        return self.__wrapped__[key]

    @property
    def name(self) -> str:
        """Get the name of the module."""
        return self._self_name

    @property
    def device(self) -> torch.device:
        """Get the device of the module.

        Returns:
            The device the module is using.
        """
        # Note: collect device from tuned module (_wrapper) to indicate correct device when loading HF pipeline.
        if self._self_state == ModuleState.TUNED:
            return self._self_wrapper.device

        # When module is not tuned, return the device of the original module (__wrapped__)
        return get_module_device(self.__wrapped__)

    @property
    def graph_specs(self) -> list[GraphSpec]:
        """Multi-graphs of the module."""
        if self._self_state == ModuleState.RECORDING:
            wrapper = cast(RecordingModule, self._self_wrapper)
            return wrapper.graph_specs
        elif self._self_state == ModuleState.PASSTHROUGH and self._self_prev_recording:
            return self._self_prev_recording.graph_specs
        elif self._self_state == ModuleState.TUNED and self._self_prev_recording:
            return self._self_prev_recording.graph_specs
        else:
            raise ValueError("Module has not recorded any samples. Cannot collect graph specs.")

    @property
    def state(self) -> ModuleState:
        """Get the state of the module."""
        return self._self_state

    @property
    def module(self):
        """Get the backends of the module."""
        return self._self_wrapper

    def enable_passthrough(self, force: bool = False):
        """Enables passthrough mode.

        Args:
            force: if True, force the module to be in the passthrough mode.
        """
        if self._self_state == ModuleState.TUNED:
            if force:
                self._reset()
                self.enable_passthrough()
            else:
                raise RuntimeError("Module is already tuned. Use force=True to reset tuned module.")
        elif self._self_state == ModuleState.RECORDING:
            self._self_prev_recording = self._self_wrapper

        self._deactivate_wrapper()
        self._self_state = ModuleState.PASSTHROUGH
        self._self_wrapper = PassthroughModule(
            self.__wrapped__,
            self._self_device,
        )

    def enable_recording(self, force: bool = False):
        """Enables recording mode.

        Args:
            force: if True, force the module to be in the recording mode.
        """
        if self._self_state == ModuleState.TUNED:
            if force:
                self._reset()
                self.enable_recording()
            else:
                raise RuntimeError("Module is already tuned. Use force=True to reset tuned module.")
        elif self._self_state in [ModuleState.PASSTHROUGH, ModuleState.INIT]:
            self._deactivate_wrapper()
            self._self_state = ModuleState.RECORDING
            if self._self_prev_recording is None:
                self._self_wrapper = RecordingModule(
                    self.__wrapped__,
                    self._self_name,
                )
            else:
                # continue with previous recording
                self._self_wrapper = self._self_prev_recording
                self._self_prev_recording = None

    def activate(self):
        """Activates the module backends."""
        if self._self_state == ModuleState.TUNED:
            self._activate_wrapper()
        elif self._self_state == ModuleState.PASSTHROUGH:
            return
        else:
            raise RuntimeError("Module is not tuned. Cannot activate backends.")

    def deactivate(self):
        """Deactivates the module backends."""
        if self._self_state == ModuleState.TUNED:
            wrapper = cast(TunedModule, self._self_wrapper)
            wrapper.deactivate()

    @staticmethod
    def is_state_dict_valid(state_dict: dict):
        """Check if the state dictionary has a wrapper module."""
        return state_dict and Module.__name__ in state_dict[Module.TYPE_KEY]

    @staticmethod
    def from_dict(module: torch.nn.Module, state_dict: dict, device: torch.device | None = None):
        """Create a wrapper module from a state dictionary."""
        if not Module.is_state_dict_valid(state_dict):
            raise ValueError(f"Invalid dictionary format for {Module.__class__.__name__}")

        self = Module(module, state_dict[Module.NAME_KEY])
        self._self_state = ModuleState.TUNED
        self._self_wrapper = TunedModule.from_dict(module, state_dict[Module.TUNED_MODULE_KEY])
        self._deploy_wrapper(device)
        return self

    def to_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """Convert the wrapper module to a state dictionary.

        The method follows torch.state_dict format so that it could be called recursively from torch.nn.Module.state_dict.
        The name of the function follows aitune convention of from/to dict. Special alias is created to match torch
        convention i.e. state_dict.
        """
        if self._self_state != ModuleState.TUNED:
            raise RuntimeError("Module is not tuned. Cannot save state_dict.")
        destination = OrderedDict() if destination is None else destination

        destination[prefix] = {
            self.TYPE_KEY: Module.__name__,  # self.__class__ returns wrapped class, use direct class object instead
            self.NAME_KEY: self._self_name,
            self.TUNED_MODULE_KEY: self._self_wrapper.to_dict(),  # pytype: disable=attribute-error
        }
        return destination

    state_dict = to_dict  # alias required by torch module.state_dict recursive call

    def tune(
        self,
        device: torch.device | None = None,
        strategy: TuneStrategy | None = None,
        dry_run: bool = False,
    ):
        """Tunes the module.

        Args:
            device: the device to use for tuning
            strategy: the tuning strategy to use
            dry_run: if True, only dry run the tuning

        Note:
            if module has already defined strategy/strategies those will take precedence over the provided one.
        """
        if self._self_state == ModuleState.TUNED:
            raise ValueError(f"Module: '{self._self_name}' has already been tuned. Reset it to do it again.")
        elif self._self_state == ModuleState.INIT or self._self_state == ModuleState.PASSTHROUGH:
            raise ValueError(f"Module: '{self._self_name}' has not recorded any samples. Cannot tune it.")

        device = device or self._self_device

        recording = cast(RecordingModule, self._self_wrapper)
        backends: OrderedDict[SampleMetadata, Backend] = OrderedDict()
        strategies = self._get_strategies_for_graph_specs(strategy, recording.graph_specs, dry_run)

        for strategy, graph_spec in zip(strategies, recording.graph_specs, strict=True):
            cache_dir = self._create_graph_cache_dir(graph_spec)

            data = recording.samples_for_graph_spec(graph_spec)
            if dry_run:
                strategy.tune_dry_run(self.__wrapped__, self._self_name, graph_spec, data, device, cache_dir)
            else:
                try:
                    self._restore_original_forward()
                    backends[graph_spec.input_spec] = strategy.tune(
                        self.__wrapped__, self._self_name, graph_spec, data, device, cache_dir
                    )
                finally:
                    self._proxy_forward()

        if not dry_run:
            self._self_prev_recording = recording
            self._self_state = ModuleState.TUNED
            self._self_wrapper = TunedModule(backends, module_name=self._self_name)
            self._offload(backends)

    def _create_graph_cache_dir(self, graph_spec: GraphSpec) -> Path:
        """Create a cache directory for the graph."""
        cache_dir = aitune_cache_dir() / self._self_name / graph_spec.name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _activate_wrapper(self):
        try:
            # revert original forward for activation which may result in jit compilation
            self._restore_original_forward()
            wrapper = cast(TunedModule, self._self_wrapper)
            wrapper.activate()
        finally:
            self._proxy_forward()

    def _deactivate_wrapper(self):
        if self._self_wrapper is not None:
            if self._self_state == ModuleState.TUNED:
                wrapper = cast(TunedModule, self._self_wrapper)
                wrapper.deactivate()
            self._self_wrapper = None
            gc.collect()

    def _deploy_wrapper(self, device: torch.device | None):
        try:
            # revert original forward for deployment which may result in jit compilation
            self._restore_original_forward()
            wrapper = cast(TunedModule, self._self_wrapper)
            wrapper.deploy(device=device)
        finally:
            self._proxy_forward()

    def _forward(self, wrapped, instance, args, kwargs) -> Any:
        """Calls one of the wrappers depending on the module state.

        Before calling a particular wrapper, the forward method is restored so that we
        avoid infinite recursion. After the implementation handles the call forward is
        restored back to point to this _forward to intercept subsequent calls.

        Args:
            wrapped: The wrapped module.
            instance: The instance of the module.
            args: The arguments to pass to the module.
            kwargs: The keyword arguments to pass to the module.

        Returns:
            The result of the call.

        Note:
            Method signature contains wrapped, instance - this is required by wrapt.decorator.
        """
        if self._self_state == ModuleState.INIT:
            self.enable_recording()
        try:
            self._restore_original_forward()
            result = self._self_wrapper(*args, **kwargs)
        finally:
            self._proxy_forward()

        return result

    def _proxy_forward(self):
        """Proxy the forward calls.

        We need to re-enable hooks, so that they will be called before and after proxied forward.
        """
        self.__wrapped__._forward_pre_hooks = self._original_forward_pre_hooks
        self.__wrapped__.forward = self._self_proxy_forward
        self.__wrapped__._forward_hooks = self._original_forward_hooks

    def _restore_original_forward(self):
        """Restore the original forward and hooks.

        We need to disable hooks, otherwise they will be called twice.
        """
        self.__wrapped__._forward_pre_hooks = OrderedDict()
        self.__wrapped__.forward = self._self_orig_forward
        self.__wrapped__._forward_hooks = OrderedDict()

    def _reset(self):
        """Resets the module to initial state."""
        self._deactivate_wrapper()
        self._self_prev_recording = None
        self._self_state = ModuleState.INIT

    def _setup_strategies(
        self,
        strategy: TuneStrategy | None,
        strategies: StrategyList | StrategyMap | None,
    ):
        """Sets up strategy or strategy_map or strategy_list depending on input args."""
        if strategy is None and strategies is None:
            strategy = DEFAULT_STRATEGY

        if strategy is not None and strategies is not None:
            raise ValueError("Only one of strategy or strategies should be provided")
        self._self_strategy = strategy
        if strategies is None:
            self._self_strategy_map = self._self_strategy_list = None
        else:
            if isinstance(strategies, list):
                self._self_strategy_list = strategies
                self._self_strategy_map = None
            elif isinstance(strategies, dict):
                self._self_strategy_map = strategies
                self._self_strategy_list = None
            else:
                raise ValueError("strategies should be a list or a dict")

    def _get_strategies_for_graph_specs(
        self,
        strategy: TuneStrategy | None,
        graph_specs: list[GraphSpec],
        dry_run: bool,
    ) -> list[TuneStrategy]:
        """Returns strategies for given graph specs.

        Module level strategies (list/dict) take precedence over provided strategy.
        The function checks if there is sufficient strategies (list/dict).
        """
        if strategy is not None:
            return [strategy] * len(graph_specs)

        if self._self_strategy_list:
            if len(self._self_strategy_list) < len(graph_specs):
                raise RuntimeError(
                    "Not enough strategies for multi-graph. "
                    f"Expected at least {len(graph_specs)}, got {len(self._self_strategy_list)}."
                )
            return self._self_strategy_list

        if self._self_strategy_map:
            errors = []
            for graph_spec in graph_specs:
                if graph_spec.input_spec not in self._self_strategy_map:
                    errors.append(f"missing strategy for graph:{graph_spec.input_spec}")
            if errors:
                raise RuntimeError("The are following errors:\n" + "\n- ".join(errors))
            return list(self._self_strategy_map.values())

        if self._self_strategy is not None:
            return [self._self_strategy] * len(graph_specs)

        if dry_run:
            return [DummyTuneStrategy()] * len(graph_specs)

        raise RuntimeError("No strategy provided. Cannot tune module.")

    def _offload(self, backends: OrderedDict[SampleMetadata, Backend]):
        """Offload the module to the meta device only if no JIT backends are used."""
        if not backends:
            return

        if any(backend.is_jit for backend in backends.values()):
            return None

        with annotate("Offloading module to meta device"):
            offload(self.__wrapped__, device=global_config.device_after_tuning)


def get_object_name(obj: Any) -> str:
    """Get the name of an object from its module and class."""
    return f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"


def sanitize_model_name(model_name: str | None) -> str | None:
    """Sanitize model name to be used as a module name."""
    return model_name.replace("/", "_") if model_name else None
