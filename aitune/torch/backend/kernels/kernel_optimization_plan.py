# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serializable plan of selected runtime kernel providers."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from torch import nn

from aitune.torch.backend.kernels.kernel_provider import KernelProvider, kernel_provider_from_dict

PROVIDERS_KEY = "providers"


@dataclass(frozen=True)
class KernelOptimizationPlan:
    """Selected kernel providers ready to install without optimization."""

    providers: tuple[KernelProvider, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete runtime plan."""
        return {PROVIDERS_KEY: [provider.to_dict() for provider in self.providers]}

    @classmethod
    def from_dict(cls, state_dict: dict[str, Any]) -> "KernelOptimizationPlan":
        """Restore a runtime plan without running kernel optimization."""
        return cls(tuple(kernel_provider_from_dict(provider) for provider in state_dict[PROVIDERS_KEY]))

    @contextmanager
    def apply(self, module: nn.Module) -> Generator[None, None, None]:
        """Temporarily apply the plan to a module.

        This context activates providers and disables gradient calculation for
        inference. The runtime is torn down and the previous gradient state is
        restored when the context exits.

        Args:
            module: Module whose functional calls should use the selected providers.
        """
        # avoid circular import
        from aitune.torch.backend.kernels.kernel_provider_runtime import KernelProviderRuntime

        runtime = KernelProviderRuntime(module, self)
        with runtime.applied():
            yield
