# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared test utilities for functional kernel provider tests."""

from collections.abc import Callable
from typing import Any, Literal

from aitune.torch.backend.kernels import KernelUtils
from aitune.torch.backend.kernels.kernel_optimization_plan import KernelOptimizationPlan
from aitune.torch.backend.kernels.kernel_provider import KernelProvider
from aitune.torch.module.recording_module import Sample


class PreferProviderKernelUtils(KernelUtils):
    """Validate candidates normally but deterministically prefer providers."""

    def benchmark_function(
        self,
        function: Callable,
        samples: list[Sample],
        return_mode: Literal["min", "max", "mean", "median", "all"] = "mean",
        warmup: int = 25,
        repeats: int = 100,
    ) -> float:
        """Return a lower synthetic latency for a provider than for the baseline."""
        del samples, return_mode, warmup, repeats
        return 1.0 if isinstance(function, KernelProvider) else 2.0


class CountingKernelProvider(KernelProvider):
    """Count calls while delegating to a real kernel provider."""

    def __init__(self, provider: KernelProvider) -> None:
        """Wrap a provider and initialize its call counter."""
        super().__init__()
        self.provider = provider
        self.calls = 0

    @property
    def supported_function(self) -> str:
        """Return the wrapped provider's supported function."""
        return self.provider.supported_function

    @property
    def name(self) -> str:
        """Return the wrapped provider's description."""
        return self.provider.name

    def _prepare(self, samples: list[Sample]) -> bool:
        """Prepare the wrapped provider."""
        return self.provider.prepare(samples)

    def _infer(self, *args: Any, **kwargs: Any) -> Any:
        """Count and delegate a provider call."""
        self.calls += 1
        return self.provider(*args, **kwargs)

    def _to_dict(self) -> dict[str, Any]:
        """Reject serialization because this wrapper is only used during tests."""
        raise NotImplementedError("Test-only counting provider is not serializable")

    @classmethod
    def _from_dict(cls, state_dict: dict[str, Any]) -> "CountingKernelProvider":
        """Reject restoration because this wrapper is only used during tests."""
        raise NotImplementedError("Test-only counting provider is not restorable")


def selected_counting_provider(plan: KernelOptimizationPlan) -> CountingKernelProvider:
    """Return the single selected counting provider and reset its call count."""
    assert len(plan.providers) == 1
    provider = plan.providers[0]
    assert isinstance(provider, CountingKernelProvider)
    provider.calls = 0
    return provider


def assert_provider_was_used(provider: CountingKernelProvider) -> None:
    """Assert that optimized inference called the selected provider."""
    assert provider.calls > 0
