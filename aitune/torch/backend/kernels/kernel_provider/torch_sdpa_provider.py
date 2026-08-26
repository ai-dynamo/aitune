# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch SDPA kernel provider."""

import logging
from typing import Any

from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention

from aitune.torch.backend.kernels.kernel_provider.kernel_provider import SDPA_FUNCTION, KernelProvider
from aitune.torch.module.sample_store import Sample

logger = logging.getLogger(__name__)


class TorchSDPAKernelProvider(KernelProvider):
    """Run PyTorch SDPA under a selected backend context."""

    def __init__(
        self,
        backend: SDPBackend = SDPBackend.FLASH_ATTENTION,
    ) -> None:
        """Initialize the selected PyTorch SDPA backend configuration."""
        super().__init__()
        self.backend = backend

    @property
    def supported_function(self) -> str:
        """Name of the function replaced by this provider."""
        return SDPA_FUNCTION

    @property
    def name(self) -> str:
        """Return a human-readable provider name."""
        return f"PyTorch SDPA {self.backend.name}"

    def _prepare(self, samples: list[Sample]) -> bool:
        """Check whether the selected PyTorch SDPA backend supports every sample."""
        with sdpa_kernel(self.backend):
            for args, kwargs in samples:
                try:
                    scaled_dot_product_attention(*args, **kwargs)
                except Exception as error:  # noqa: BLE001 - an unsupported backend may raise any exception
                    logger.debug("PyTorch SDPA %s not supported due to: %s", self.backend.name, error)
                    return False
        return True

    def _infer(self, *args: Any, **kwargs: Any) -> Any:
        """Run SDPA under the configured backend context."""
        with sdpa_kernel(self.backend):
            return scaled_dot_product_attention(*args, **kwargs)

    def _to_dict(self) -> dict[str, Any]:
        """Serialize the selected backend configuration."""
        return {
            "backend": self.backend.name,
        }

    @classmethod
    def _from_dict(cls, state_dict: dict[str, Any]) -> "TorchSDPAKernelProvider":
        """Restore the selected backend configuration."""
        return cls(backend=getattr(SDPBackend, state_dict["backend"]))
