# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SageAttention kernel provider."""

import logging
from collections.abc import Callable
from functools import cached_property, lru_cache
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from typing import Any

import torch

from aitune.torch.backend.kernels.kernel_provider.kernel_provider import SDPA_FUNCTION, KernelProvider
from aitune.torch.module.recording_module import Sample

logger = logging.getLogger(__name__)

_RUNTIME_UNAVAILABLE_MESSAGE = (
    "SageAttention is not available. Install it with `pip install sageattention`. "
    "Note: The PyPI package provides SageAttention V1, which uses Triton. "
    "For best performance, install the latest version from source."
)


@lru_cache(maxsize=1)
def _sageattention_version() -> str:
    """Return the installed SageAttention version."""
    try:
        sageattention = import_module("sageattention")
    except ImportError:
        return "cannot be imported"
    version = getattr(sageattention, "__version__", None)
    if version is not None:
        return f"v{version}"
    try:
        return f"v{pkg_version('sageattention')}"
    except PackageNotFoundError:
        return "unknown version"


class SageAttentionKernelProvider(KernelProvider):
    """Run SageAttention with a sample-derived argument and layout plan."""

    def __init__(self) -> None:
        """Initialize SageAttention with a default, not-yet-ready inference plan."""
        super().__init__()
        self.needs_kwargs_mapping = False
        self.use_diffusers_native_hnd_view = False

    @cached_property
    def _backend(self) -> Callable[..., Any]:
        """Load and validate the SageAttention runtime function lazily."""
        try:
            backend = import_module("sageattention")
        except ImportError as error:
            raise RuntimeError(_RUNTIME_UNAVAILABLE_MESSAGE) from error

        sageattn = getattr(backend, "sageattn", None)
        if not callable(sageattn):
            raise RuntimeError(_RUNTIME_UNAVAILABLE_MESSAGE)
        return sageattn

    @property
    def supported_function(self) -> str:
        """Name of the function replaced by this provider."""
        return SDPA_FUNCTION

    @property
    def name(self) -> str:
        """Return a human-readable provider name."""
        return f"Sage Attention {_sageattention_version()}"

    def _prepare(self, samples: list[Sample]) -> bool:
        """Select one inference plan for all representative samples."""
        replacement_plans = {_sample_replacement_plan(args, kwargs) for args, kwargs in samples}
        if len(replacement_plans) != 1:
            logger.debug("SageAttention requires consistent representative samples")
            return False

        replacement_plan = replacement_plans.pop()
        self.needs_kwargs_mapping, self.use_diffusers_native_hnd_view = replacement_plan
        return True

    def _infer(self, *args: Any, **kwargs: Any) -> Any:
        """Run the selected SageAttention inference plan."""
        kwargs = _normalize_sage_kwargs(kwargs)
        if self.needs_kwargs_mapping:
            _query_key_value_to_qkv(kwargs)

        if self.use_diffusers_native_hnd_view:
            kwargs["q"] = kwargs["q"].transpose(1, 2)
            kwargs["k"] = kwargs["k"].transpose(1, 2)
            kwargs["v"] = kwargs["v"].transpose(1, 2)
            output = self._backend(**kwargs, tensor_layout="NHD")
            return output.transpose(1, 2)

        return self._backend(*args, **kwargs)

    def _to_dict(self) -> dict[str, Any]:
        """Serialize the selected inference plan."""
        return {
            "needs_kwargs_mapping": self.needs_kwargs_mapping,
            "use_diffusers_native_hnd_view": self.use_diffusers_native_hnd_view,
        }

    @classmethod
    def _from_dict(cls, state_dict: dict[str, Any]) -> "SageAttentionKernelProvider":
        """Restore the selected inference plan."""
        provider = cls()
        provider.needs_kwargs_mapping = state_dict["needs_kwargs_mapping"]
        provider.use_diffusers_native_hnd_view = state_dict["use_diffusers_native_hnd_view"]
        return provider


def _sample_replacement_plan(args, kwargs) -> tuple[bool, bool]:
    """Return the inference plan required by one sample."""
    return _needs_mapping_of_kwargs(args, kwargs), _is_diffusers_native_hnd_sample(args, kwargs)


def _needs_mapping_of_kwargs(args, kwargs) -> bool:
    """Return whether PyTorch SDPA keyword names must be mapped to SageAttention names."""
    return len(args) == 0 and "q" not in kwargs and "query" in kwargs


def _is_diffusers_native_hnd_sample(args, kwargs) -> bool:
    """Check the representative sample for the Diffusers native-dispatch HND view pattern."""
    if args:
        return False

    sample_kwargs = kwargs
    if _needs_mapping_of_kwargs(args, kwargs):
        sample_kwargs = dict(kwargs)
        _query_key_value_to_qkv(sample_kwargs)

    q = sample_kwargs.get("q")
    k = sample_kwargs.get("k")
    v = sample_kwargs.get("v")
    return _is_diffusers_native_hnd_view(q) and _is_diffusers_native_hnd_view(k) and _is_diffusers_native_hnd_view(v)


def _is_diffusers_native_hnd_view(tensor) -> bool:
    """Check for an HND tensor that is a cheap transpose view of contiguous NHD data."""
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4 or tensor.stride(-1) != 1:
        return False

    _, num_heads, _, head_dim = tensor.shape
    return tensor.stride(1) == head_dim and tensor.stride(2) == num_heads * head_dim


def _query_key_value_to_qkv(kwargs) -> None:
    """Map query, key and value keyword arguments to q, k and v."""
    kwargs["q"] = kwargs.pop("query")
    kwargs["k"] = kwargs.pop("key")
    kwargs["v"] = kwargs.pop("value")


def _normalize_sage_kwargs(kwargs):
    """Map SDPA keyword names to SageAttention keyword names."""
    kwargs = dict(kwargs)
    if "scale" in kwargs and "sm_scale" not in kwargs:
        kwargs["sm_scale"] = kwargs.pop("scale")
    return kwargs
