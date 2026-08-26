# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contracts for asynchronous kernel generation."""

from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass

from aitune.torch.backend.kernels.kernel_provider.kernel_provider import KernelProvider
from aitune.torch.module.sample_store import Sample


@dataclass(frozen=True)
class KernelGenerationResult:
    """Result of generating one provider for one PyTorch function.

    A successful result contains a serializable provider. A controlled
    generation failure contains an error message instead. Unexpected failures
    may still be raised by the corresponding ``Future``.
    """

    function: str
    provider: KernelProvider | None
    description: str
    error: str | None = None

    def __post_init__(self) -> None:
        """Require exactly one success or controlled-failure outcome."""
        if (self.provider is None) == (self.error is None):
            raise ValueError("Kernel generation result must contain exactly one of a provider or an error")
        if self.provider is not None and not isinstance(self.provider, KernelProvider):
            raise TypeError("Kernel generation result provider must be a KernelProvider")

    @property
    def succeeded(self) -> bool:
        """Whether generation produced a usable provider."""
        return self.provider is not None and self.error is None


class KernelGenerator(ABC):
    """Asynchronously generate kernel providers for supported functions."""

    @abstractmethod
    def __repr__(self) -> str:
        """Return a human-readable kernel generator description."""
        ...

    @abstractmethod
    def supports_functions(self) -> list[str]:
        """List the PyTorch function names supported by the generator."""
        ...

    @abstractmethod
    def prepare(self, function: str, samples: list[Sample]) -> bool:
        """Prepare generation and return whether all samples are supported."""
        ...

    @abstractmethod
    def submit(self, function: str, samples: list[Sample]) -> Future[KernelGenerationResult]:
        """Start one generation task without waiting for it to finish."""
        ...
