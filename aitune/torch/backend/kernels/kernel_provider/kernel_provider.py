# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel provider interface."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, TypeVar

from aitune.torch.module.recording_module import Sample

TYPE_KEY = "type"
SDPA_FUNCTION = "scaled_dot_product_attention"  # type: ignore[assignment]

KernelProviderType = TypeVar("KernelProviderType", bound="KernelProvider")
_KERNEL_PROVIDER_TYPES: dict[str, type["KernelProvider"]] = {}


class KernelProviderState(Enum):
    """Kernel provider lifecycle state."""

    INIT = "init"
    READY = "ready"


class KernelProvider(ABC):
    """Kernel provider interface.

    Allows preparing a kernel given list of samples and doing inference with the it.

    At the beginning it is in INIT state, after preparing it is in READY state. Once prepared for inference
    it can be serialized and restored with ``to_dict`` and ``from_dict`` respectively.

    Each provider supports a single function (from torch.nn.functional).
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register each provider class for state-dict deserialization."""
        super().__init_subclass__(**kwargs)
        provider_type = cls.__name__
        registered_class = _KERNEL_PROVIDER_TYPES.get(provider_type)
        if registered_class is not None and registered_class is not cls:
            raise ValueError(f"Kernel provider type is already registered: {provider_type}")
        _KERNEL_PROVIDER_TYPES[provider_type] = cls

    def __init__(self) -> None:
        """Initialize a kernel provider that is not ready for inference."""
        self.state = KernelProviderState.INIT

    @property
    @abstractmethod
    def supported_function(self) -> str:
        """Name of the ``torch.nn.functional`` function replaced by this provider."""
        ...

    @property
    def name(self) -> str:
        """Return the kernel provider name."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        """Return a human-readable kernel provider description."""
        return self.name

    def prepare(self: KernelProviderType, samples: list[Sample]) -> bool:
        """Prepare this provider for inference using representative samples.

        Args:
            samples: Representative calls of :attr:`supported_function`.

        Returns:
            True when every sample is supported, otherwise False.
        """
        if self.state is KernelProviderState.READY:
            return True

        if not self._prepare(samples):
            return False

        self.state = KernelProviderState.READY
        return True

    @abstractmethod
    def _prepare(self, samples: list[Sample]) -> bool:
        """Validate samples and populate the state required for inference."""
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run inference using this provider."""
        if self.state is not KernelProviderState.READY:
            raise RuntimeError(f"Kernel provider {self.name} must be prepared before inference")
        return self._infer(*args, **kwargs)

    @abstractmethod
    def _infer(self, *args: Any, **kwargs: Any) -> Any:
        """Run provider-specific inference."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize the state required to restore this provider."""
        if self.state is not KernelProviderState.READY:
            raise RuntimeError(f"Kernel provider {self.name} must be prepared before serialization")
        return {TYPE_KEY: self.__class__.__name__, **self._to_dict()}

    @abstractmethod
    def _to_dict(self) -> dict[str, Any]:
        """Serialize provider-specific inference state."""
        ...

    @classmethod
    def from_dict(
        cls: type[KernelProviderType],
        state_dict: dict[str, Any],
    ) -> KernelProviderType:
        """Restore a provider that is ready for inference."""
        provider_type = state_dict.get(TYPE_KEY)
        if provider_type != cls.__name__:
            raise ValueError(f"Invalid kernel provider type for {cls.__name__}: {provider_type}")

        provider = cls._from_dict(state_dict)
        provider.state = KernelProviderState.READY
        return provider

    @classmethod
    @abstractmethod
    def _from_dict(
        cls: type[KernelProviderType],
        state_dict: dict[str, Any],
    ) -> KernelProviderType:
        """Restore provider-specific inference state."""
        ...


def kernel_provider_from_dict(state_dict: dict[str, Any]) -> KernelProvider:
    """Restore the concrete provider identified by its serialized class name."""
    provider_type = state_dict[TYPE_KEY]
    provider_class = _KERNEL_PROVIDER_TYPES.get(provider_type)
    if provider_class is None:
        raise ValueError(f"Unknown kernel provider type: {provider_type}")
    return provider_class.from_dict(state_dict)
