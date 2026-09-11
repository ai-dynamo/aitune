# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Composite backend selecting kernel providers before delegating execution."""

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch.nn as nn
from torch.nn.attention import SDPBackend

from aitune.torch.backend.backend import (
    Backend,
    BackendConfig,
    BackendState,
    BuildMode,
    ExecutionMode,
)
from aitune.torch.backend.torch_inductor_jit_backend import TorchInductorJitBackend
from aitune.torch.distributed import coordinator
from aitune.torch.kernel_forge.kernel_optimization_plan import KernelOptimizationPlan
from aitune.torch.kernel_forge.kernel_optimizer import KernelOptimizer
from aitune.torch.kernel_forge.kernel_provider import (
    DiffusersAttentionBackend,
    DiffusersAttentionKernelProvider,
    FlashAttention4KernelProvider,
    KernelGenerator,
    KernelProvider,
    TorchSDPAKernelProvider,
)
from aitune.torch.kernel_forge.kernel_provider_runtime import KernelProviderRuntime
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import SampleStore
from aitune.torch.utils.module import move_module_to_device
from aitune.utils.env_vars import AITUNE_KERNEL_GENERATION_TIMEOUT
from aitune.utils.validation import in_range


def _default_kernel_providers() -> list[KernelProvider]:
    """Create the built-in attention providers used by the default backend."""
    return [
        TorchSDPAKernelProvider(SDPBackend.CUDNN_ATTENTION),
        # Flash Attention
        TorchSDPAKernelProvider(SDPBackend.FLASH_ATTENTION),  # v2
        DiffusersAttentionKernelProvider(DiffusersAttentionBackend.FLASH_3_HUB),  # v3
        FlashAttention4KernelProvider(),  # v4
    ]


@dataclass
class KernelSelectorBackendConfig(BackendConfig):
    """Configuration for kernel selection performed before delegate build.

    This configuration is meant to be serialized (``to_dict``) but not deserialized (``from_dict``).
    The corresponding backend needs this configuration during ``build`` only. Once backend is built,
    it can serialize optimization plan instead of this configuration object.

    Args:
        kernel_providers: Static providers considered during kernel optimization. Defaults to PyTorch Flash
            Attention, PyTorch cuDNN Attention, and FlashAttention-4 providers.
        kernel_generators: Asynchronous generators considered during kernel optimization.
        provider_min_time_share_percent: Minimum profiled time share required for static providers.
        generator_min_time_share_percent: Minimum profiled time share required for dynamic generators.
        generation_timeout: Maximum seconds to wait for submitted generators.
    """

    kernel_providers: KernelProvider | list[KernelProvider] | None = field(default_factory=_default_kernel_providers)
    kernel_generators: KernelGenerator | list[KernelGenerator] | None = None
    provider_min_time_share_percent: float = 0.0
    generator_min_time_share_percent: float = 10.0
    generation_timeout: float = AITUNE_KERNEL_GENERATION_TIMEOUT

    def __post_init__(self) -> None:
        """Normalize kernel sources and validate threshold values."""
        self.kernel_providers = self._normalize_providers(self.kernel_providers)
        self.kernel_generators = self._normalize_generators(self.kernel_generators)
        if not self.kernel_providers and not self.kernel_generators:
            raise ValueError("At least one kernel provider or generator must be provided")

        in_range(
            self.provider_min_time_share_percent,
            min_value=0,
            max_value=100,
            name="provider_min_time_share_percent",
        )
        in_range(
            self.generator_min_time_share_percent,
            min_value=0,
            max_value=100,
            name="generator_min_time_share_percent",
        )
        if self.generation_timeout < 0:
            raise ValueError("generation_timeout must be greater than or equal to 0")

    @staticmethod
    def _normalize_providers(
        providers: KernelProvider | list[KernelProvider] | None,
    ) -> list[KernelProvider]:
        """Normalize and validate static kernel providers."""
        if providers is None:
            return []
        if isinstance(providers, KernelProvider):
            return [providers]
        return providers

    @staticmethod
    def _normalize_generators(
        generators: KernelGenerator | list[KernelGenerator] | None,
    ) -> list[KernelGenerator]:
        """Normalize and validate asynchronous kernel generators."""
        if generators is None:
            return []
        if isinstance(generators, KernelGenerator):
            return [generators]
        return generators

    def describe(self) -> str:
        """Return a human-readable selector configuration."""
        provider_descriptions = [str(provider) for provider in self.kernel_providers]
        generator_descriptions = [str(generator) for generator in self.kernel_generators]
        return (
            f"kernel_providers={provider_descriptions},"
            f"kernel_generators={generator_descriptions},"
            f"provider_min_time_share_percent={self.provider_min_time_share_percent},"
            f"generator_min_time_share_percent={self.generator_min_time_share_percent},"
            f"generation_timeout={self.generation_timeout}"
        )

    @classmethod
    def from_dict(cls, _data: dict) -> "KernelSelectorBackendConfig":
        """Reject deserialization of one-way kernel source metadata."""
        raise NotImplementedError(f"{cls.__name__} supports one-way serialization only")

    def to_dict(self) -> dict[str, Any]:
        """Return one-way, JSON-serializable cache and diagnostic metadata.

        :meth:`KernelProvider.to_dict` is intentionally not used here. Provider
        serialization captures prepared runtime state and requires the provider to
        be ready, whereas backend configuration is keyed and saved before providers
        are prepared i.e. before ``build`` is called. Selected ready providers are
        serialized separately in :class:`KernelOptimizationPlan` checkpoints.
        """
        return {
            "kernel_providers": [provider.name for provider in self.kernel_providers],
            "kernel_generators": [generator.name for generator in self.kernel_generators],
            "provider_min_time_share_percent": self.provider_min_time_share_percent,
            "generator_min_time_share_percent": self.generator_min_time_share_percent,
            "generation_timeout": self.generation_timeout,
        }


class KernelSelectorBackend(Backend):
    """Select kernel providers and apply their plan while building a delegate backend.

    The backend adopts the delegate's build mode and supported execution modes. A JIT delegate keeps the selected
    provider plan active at runtime and serializes it in checkpoints. An AOT delegate captures the applied replacements
    while building its artifact, after which the live provider runtime is discarded.
    """

    # Instance values are narrowed to the delegate's declared modes in __init__ and from_dict.
    _build_mode = BuildMode.JUST_IN_TIME
    _execution_modes = frozenset({ExecutionMode.SINGLE_GPU, ExecutionMode.MULTI_GPU})

    STATE_TYPE = "type"
    STATE_VERSION = "version"
    STATE_DEVICE = "device"
    STATE_KERNEL_PLAN = "kernel_plan"
    STATE_DELEGATE_BACKEND = "delegate_backend"
    state_version = 1

    def __init__(
        self,
        config: KernelSelectorBackendConfig | None = None,
        delegate_backend: Backend | None = None,
    ) -> None:
        """Initialize the composite backend.

        Args:
            config: Kernel providers, generators, and selection thresholds used before the delegate builds. Defaults
                to :class:`KernelSelectorBackendConfig` with its built-in attention providers.
            delegate_backend: Backend that compiles or executes the module after provider selection. Defaults to
                :class:`TorchInductorJitBackend`.
        """
        super().__init__()
        self._config = config if config is not None else KernelSelectorBackendConfig()
        self._delegate_backend = delegate_backend if delegate_backend is not None else TorchInductorJitBackend()
        self._runtime: KernelProviderRuntime | None = None
        self._adopt_delegate_modes()

    def key(self) -> str:
        """Return a cache key combining selector and delegate configuration."""
        if self._config is None:
            return f"{self.__class__.__name__}_{self._delegate_backend.key()}"
        return f"{self.__class__.__name__}_{self._config.key()}_{self._delegate_backend.key()}"

    def describe(self) -> str:
        """Describe selector and delegate configuration."""
        if self._config is None:
            return f"{self.__class__.__name__}(delegate={self._delegate_backend.describe()})"
        return f"{self.__class__.__name__}({self._config.describe()},delegate={self._delegate_backend.describe()})"

    def _build(
        self,
        module: nn.Module,
        graph_spec: GraphSpec,
        samples: SampleStore,
        cache_dir: Path,
    ) -> Backend:
        """Select providers and build the delegate while the plan is applied."""
        move_module_to_device(module, self._device)

        optimizer = self._create_optimizer()
        plan = optimizer.make_plan(module, samples, module=module)
        plan = self._synchronize_plan(plan)
        runtime = KernelProviderRuntime(module, plan)

        delegate_cache_dir = cache_dir / "delegate_backend"
        delegate_cache_dir.mkdir(parents=True, exist_ok=True)

        with runtime.applied():
            self._delegate_backend = self._delegate_backend.build(
                module, graph_spec, samples, self._device, delegate_cache_dir
            )

        if self._build_mode is BuildMode.JUST_IN_TIME:
            runtime.activate()  # leave backend fully active after build
            self._runtime = runtime

        return self

    def _synchronize_plan(
        self,
        plan: KernelOptimizationPlan,
    ) -> KernelOptimizationPlan:
        """Apply rank zero's serialized optimization plan to every distributed worker.

        Ranks can have slightly different plans due to timing variations of the providers.
        This is because torch functions are low level non distributed functions. There is no
        coordination between ranks, hance there is no guarantee that all ranks will have the same plan.
        The assumption here is that rank zero's plan is the winning plan.

        Args:
            plan: The plan to synchronize.

        Returns:
            The synchronized plan.
        """
        with coordinator.raise_if_any_rank_fails("Serializing kernel optimization plans"):
            local_plan_state = plan.to_dict()

        rank_zero_plan_state = coordinator.broadcast_from_rank0(local_plan_state)
        with coordinator.raise_if_any_rank_fails("Restoring rank-zero kernel optimization plan"):
            synchronized_plan = KernelOptimizationPlan.from_dict(rank_zero_plan_state)

        self._build_results.append(
            {
                "detailed_build_info": {
                    "local_optimization_plan": local_plan_state,
                    "synchronized_plan": rank_zero_plan_state,
                }
            },
        )
        return synchronized_plan

    def _create_optimizer(self) -> KernelOptimizer:
        """Create the kernel optimizer from backend configuration."""
        return KernelOptimizer(
            kernel_providers=self._config.kernel_providers,
            kernel_generators=self._config.kernel_generators,
            provider_min_time_share_percent=self._config.provider_min_time_share_percent,
            generator_min_time_share_percent=self._config.generator_min_time_share_percent,
            generation_timeout=self._config.generation_timeout,
        )

    def _activate(self) -> None:
        """Activate the runtime and delegate."""
        if self._runtime is not None:
            self._runtime.activate()
        try:
            self._delegate_backend.activate()
        except Exception:
            # leave consistent state after exception
            if self._runtime is not None:
                self._runtime.deactivate()
            raise

    def _infer(self, *args: Any, **kwargs: Any) -> Any:
        """Run inference through the delegate backend."""
        return self._delegate_backend.infer(*args, **kwargs)

    def _deactivate(self) -> None:
        """Deactivate the delegate and runtime."""
        self._delegate_backend.deactivate()
        if self._runtime is not None:
            self._runtime.deactivate()

    def _deploy(self) -> None:
        """Activate the runtime and deploy the restored delegate."""
        if self._runtime is not None:
            self._runtime.activate()
        try:
            self._delegate_backend.deploy(self._device)
        except Exception:
            if self._runtime is not None:
                self._runtime.deactivate()
            raise
        gc.collect()

    def to_dict(self) -> dict[str, Any]:
        """Return state required to restore the selected plan and delegate."""
        state = {
            self.STATE_TYPE: self.__class__.__name__,
            self.STATE_VERSION: self.state_version,
            self.STATE_DEVICE: self._device,
            self.STATE_DELEGATE_BACKEND: self._delegate_backend.to_dict(),
        }
        if self._build_mode is BuildMode.JUST_IN_TIME:
            if self._runtime is None:
                raise RuntimeError("Kernel provider runtime is not initialized")
            state[self.STATE_KERNEL_PLAN] = self._runtime.plan.to_dict()

        return state

    @classmethod
    def from_dict(cls, module: nn.Module | None, state_dict: dict) -> "KernelSelectorBackend":
        """Restore the selected plan and delegate without running optimization."""
        if state_dict.get(cls.STATE_TYPE) != cls.__name__:
            raise ValueError(f"Invalid state_dict type: {state_dict.get(cls.STATE_TYPE)}")
        if state_dict.get(cls.STATE_VERSION) != cls.state_version:
            raise ValueError(f"Unsupported KernelSelectorBackend state version: {state_dict.get(cls.STATE_VERSION)}")

        delegate_state = state_dict.get(cls.STATE_DELEGATE_BACKEND)
        if not isinstance(delegate_state, dict):
            raise ValueError("KernelSelectorBackend state requires a delegate backend")
        delegate_class = cls._delegate_class_from_state(delegate_state)
        delegate_backend = delegate_class.from_dict(module, delegate_state)

        backend = cls.__new__(cls)
        Backend.__init__(backend)
        backend._delegate_backend = delegate_backend
        backend._runtime = None
        backend._adopt_delegate_modes()
        backend._set_device(state_dict.get(cls.STATE_DEVICE))

        if backend.build_mode is BuildMode.JUST_IN_TIME:
            if module is None:
                raise ValueError("Module is required to restore a just-in-time delegate")
            plan_state = state_dict.get(cls.STATE_KERNEL_PLAN)
            plan = KernelOptimizationPlan.from_dict(plan_state)
            backend._runtime = KernelProviderRuntime(module, plan)

        backend.state = BackendState.CHECKPOINT_LOADED
        return backend

    @staticmethod
    def _delegate_class_from_state(state_dict: dict[str, Any]) -> type[Backend]:
        """Resolve a serialized delegate backend by its class name."""
        delegate_type = state_dict.get("type")
        if not isinstance(delegate_type, str):
            raise ValueError("Delegate backend state requires a string type")

        backend_classes = {
            backend_class.__name__: backend_class
            for backend_class in KernelSelectorBackend._backend_subclasses(Backend)
        }
        try:
            return backend_classes[delegate_type]
        except KeyError as error:
            raise ValueError(f"Unknown delegate backend type: {delegate_type}") from error

    @staticmethod
    def _backend_subclasses(backend_class: type[Backend]) -> list[type[Backend]]:
        """Return all recursively defined backend subclasses."""
        subclasses = []
        for subclass in backend_class.__subclasses__():
            subclasses.append(subclass)
            subclasses.extend(KernelSelectorBackend._backend_subclasses(subclass))
        return subclasses

    def _adopt_delegate_modes(self) -> None:
        """Expose the delegate's build and execution capabilities."""
        self._build_mode = self._delegate_backend.build_mode
        self._execution_modes = self._delegate_backend._execution_modes
