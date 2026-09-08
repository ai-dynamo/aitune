# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diffusers attention dispatcher kernel provider."""

from collections.abc import Callable
from enum import Enum
from functools import cache, cached_property, partial
from importlib import import_module
from types import ModuleType
from typing import Any

import torch

from aitune.torch.backend.kernels.kernel_provider.kernel_provider import SDPA_FUNCTION, KernelProvider
from aitune.torch.module.sample_store import Sample

_RUNTIME_UNAVAILABLE_MESSAGE = (
    "Diffusers attention dispatcher is not available. Install it with `pip install 'diffusers>=0.35.0'`. "
    "Hub backends additionally require `pip install 'kernels>=0.12'`."
)


@cache
def _import_diffusers_attention_dispatch() -> ModuleType | None:
    """Import and cache the optional Diffusers attention dispatcher module."""
    try:
        return import_module("diffusers.models.attention_dispatch")
    except ImportError:
        return None


class DiffusersAttentionBackend(str, Enum):
    """Diffusers attention backends compatible with SDPA replacement."""

    FLEX = "flex"
    FLASH = "flash"
    FLASH_HUB = "flash_hub"
    FLASH_3 = "_flash_3"
    FLASH_3_HUB = "_flash_3_hub"
    FLASH_4_HUB = "flash_4_hub"
    SAGE = "sage"
    SAGE_HUB = "sage_hub"
    SAGE_QK_INT8_PV_FP8_CUDA = "_sage_qk_int8_pv_fp8_cuda"
    SAGE_QK_INT8_PV_FP8_CUDA_SM90 = "_sage_qk_int8_pv_fp8_cuda_sm90"
    SAGE_QK_INT8_PV_FP16_CUDA = "_sage_qk_int8_pv_fp16_cuda"
    SAGE_QK_INT8_PV_FP16_TRITON = "_sage_qk_int8_pv_fp16_triton"
    XFORMERS = "xformers"


class DiffusersAttentionKernelProvider(KernelProvider):
    """Run SDPA with a selected Diffusers attention dispatcher backend."""

    def __init__(self, backend: DiffusersAttentionBackend) -> None:
        """Initialize the selected Diffusers attention backend."""
        super().__init__()
        if not isinstance(backend, DiffusersAttentionBackend):
            raise TypeError("backend must be a DiffusersAttentionBackend")
        self.backend = backend

    @property
    def supported_function(self) -> str:
        """Name of the function replaced by this provider."""
        return SDPA_FUNCTION

    @property
    def name(self) -> str:
        """Human-readable provider name."""
        return f"Diffusers attention {self.backend.value}"

    @cached_property
    def _backend(self) -> Callable[..., torch.Tensor]:
        """Load the Diffusers dispatcher bound to the selected backend lazily."""
        attention_dispatch = _import_diffusers_attention_dispatch()
        if attention_dispatch is None:
            raise RuntimeError(_RUNTIME_UNAVAILABLE_MESSAGE)

        dispatch_attention_fn = getattr(attention_dispatch, "dispatch_attention_fn", None)
        attention_backend_name = getattr(attention_dispatch, "AttentionBackendName", None)
        attention_backend = getattr(attention_dispatch, "attention_backend", None)
        if (
            not callable(dispatch_attention_fn)
            or not callable(attention_backend_name)
            or not callable(attention_backend)
        ):
            raise RuntimeError(_RUNTIME_UNAVAILABLE_MESSAGE)

        try:
            backend = attention_backend_name(self.backend.value)
        except ValueError as error:
            raise RuntimeError(
                f"Diffusers attention backend {self.backend.value!r} is not available in the installed version"
            ) from error

        # Entering the context validates and initializes the backend, including downloading Hub kernels when needed.
        # Inference selects the initialized backend explicitly, so the context does not need to remain active.
        with attention_backend(backend):
            pass
        return partial(dispatch_attention_fn, backend=backend)

    def _load_runtime_dependencies(self) -> None:
        """Load Diffusers attention dependencies before graph capture."""
        _ = self._backend

    def _prepare(self, samples: list[Sample]) -> bool:
        """Prepare the provider without running inference."""
        return True

    def _infer(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        """Run SDPA through the selected Diffusers attention backend."""
        output = self._backend(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask,
            dropout_p,
            is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        return output.transpose(1, 2)

    def _to_dict(self) -> dict[str, Any]:
        """Serialize the selected Diffusers attention backend."""
        return {"backend": self.backend.value}

    @classmethod
    def _from_dict(cls, state_dict: dict[str, Any]) -> "DiffusersAttentionKernelProvider":
        """Restore the selected Diffusers attention backend."""
        return cls(backend=DiffusersAttentionBackend(state_dict["backend"]))
