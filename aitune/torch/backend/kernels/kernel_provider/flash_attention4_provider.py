# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashAttention-4 kernel provider."""

import logging
from collections.abc import Callable
from functools import cached_property, lru_cache
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from typing import Any

import torch

from aitune.torch.backend.kernels.kernel_provider.kernel_provider import SDPA_FUNCTION, KernelProvider
from aitune.torch.module.sample_store import Sample

logger = logging.getLogger(__name__)

_POSITIONAL_QKV = "positional_qkv"
_KEYWORD_QKV = "keyword_qkv"
_RUNTIME_UNAVAILABLE_MESSAGE = "FlashAttention-4 is not available. Install it with `pip install --pre flash-attn-4`."


@lru_cache(maxsize=1)
def _flash_attention4_version() -> str:
    """Return the installed FlashAttention-4 version."""
    try:
        backend = import_module("flash_attn.cute")
    except ImportError:
        return "cannot be imported"

    version = getattr(backend, "__version__", None)
    if version is not None and version != "0.0.0":
        return f"v{version}"

    for package_name in ("flash-attn-4", "fa4"):
        try:
            return f"v{pkg_version(package_name)}"
        except PackageNotFoundError:
            pass
    return "unknown version"


class FlashAttention4KernelProvider(KernelProvider):
    """Run FlashAttention-4 with a sample-derived argument and layout plan."""

    def __init__(self) -> None:
        """Initialize FlashAttention-4 with a default, not-yet-ready inference plan."""
        super().__init__()
        self.qkv_source = _POSITIONAL_QKV
        self.flash_kwargs: dict[str, Any] = {}
        self.copy_free_layout = False

    @cached_property
    def _backend(self) -> Callable[..., Any]:
        """Load and validate the FlashAttention-4 runtime function lazily."""
        try:
            backend = import_module("flash_attn.cute")
        except ImportError as error:
            raise RuntimeError(_RUNTIME_UNAVAILABLE_MESSAGE) from error

        flash_attn_func = getattr(backend, "flash_attn_func", None)
        if not callable(flash_attn_func):
            raise RuntimeError(_RUNTIME_UNAVAILABLE_MESSAGE)
        return flash_attn_func

    @property
    def supported_function(self) -> str:
        """Name of the function replaced by this provider."""
        return SDPA_FUNCTION

    @property
    def name(self) -> str:
        """Return a human-readable provider name."""
        return f"FlashAttention-4 {_flash_attention4_version()}"

    def _prepare(self, samples: list[Sample]) -> bool:
        """Select one inference plan for all representative samples."""
        try:
            replacement_plans = {_sample_replacement_plan(sample) for sample in samples}
        except ValueError as error:
            logger.debug("FlashAttention-4 not supported due to: %s", error)
            return False

        if len(replacement_plans) != 1:
            logger.debug("FlashAttention-4 requires consistent representative samples")
            return False

        qkv_source, flash_kwargs, copy_free_layout = replacement_plans.pop()
        self.qkv_source = qkv_source
        self.flash_kwargs = dict(flash_kwargs)
        self.copy_free_layout = copy_free_layout
        if not copy_free_layout:
            logger.warning(
                "FlashAttention-4 must copy query, key, and value tensors from HND to NHD layout on every call; "
                "this may reduce performance. Use HND views of NHD-contiguous tensors to enable the copy-free path."
            )
        return True

    def _infer(self, *args: Any, **kwargs: Any) -> Any:
        """Run the selected FlashAttention-4 inference plan."""
        if self.qkv_source == _POSITIONAL_QKV:
            q, k, v = args[:3]
        else:
            q, k, v = kwargs["query"], kwargs["key"], kwargs["value"]

        func = _CALL_IMPLS[self.copy_free_layout]
        return func(self._backend, q, k, v, self.flash_kwargs)

    def _to_dict(self) -> dict[str, Any]:
        """Serialize the selected inference plan."""
        return {
            "qkv_source": self.qkv_source,
            "flash_kwargs": dict(self.flash_kwargs),
            "copy_free_layout": self.copy_free_layout,
        }

    @classmethod
    def _from_dict(cls, state_dict: dict[str, Any]) -> "FlashAttention4KernelProvider":
        """Restore the selected inference plan."""
        qkv_source = state_dict["qkv_source"]
        provider = cls()
        provider.qkv_source = qkv_source
        provider.flash_kwargs = dict(state_dict["flash_kwargs"])
        provider.copy_free_layout = state_dict["copy_free_layout"]
        return provider


def _sample_replacement_plan(sample: Sample) -> tuple[str, tuple[tuple[str, Any], ...], bool]:
    """Build an inference plan for one representative sample."""
    args, kwargs = sample
    qkv_source = _qkv_source(args, kwargs)
    q, k, v, flash_kwargs = _normalize_sdpa_inputs(args, kwargs)
    copy_free_layout = all(_hnd_to_nhd_view(tensor).is_contiguous() for tensor in (q, k, v))
    return qkv_source, tuple(flash_kwargs.items()), copy_free_layout


def _normalize_sdpa_inputs(args, kwargs):
    """Normalize PyTorch SDPA arguments to FlashAttention-4 arguments."""
    kwargs = dict(kwargs)
    q, k, v = _extract_qkv(args, kwargs)

    attn_mask = kwargs.pop("attn_mask", None)
    if attn_mask is not None:
        raise ValueError("FlashAttention-4 provider does not support attn_mask")

    dropout_p = kwargs.pop("dropout_p", 0.0)
    if dropout_p != 0.0:
        raise ValueError("FlashAttention-4 provider only supports dropout_p=0.0")

    is_causal = kwargs.pop("is_causal", False)
    scale = kwargs.pop("scale", None)
    enable_gqa = kwargs.pop("enable_gqa", False)
    if kwargs:
        raise ValueError(f"FlashAttention-4 provider does not support kwargs: {sorted(kwargs)}")

    uses_gqa = _validate_qkv(q, k, v, enable_gqa=enable_gqa)
    flash_kwargs = {"causal": is_causal}
    if scale is not None:
        flash_kwargs["softmax_scale"] = scale
    if uses_gqa:
        flash_kwargs["pack_gqa"] = False
    return q, k, v, flash_kwargs


def _qkv_source(args, kwargs) -> str:
    """Return how q/k/v are passed in the representative sample."""
    if len(args) == 3:
        return _POSITIONAL_QKV
    if len(args) == 0 and {"query", "key", "value"} <= kwargs.keys():
        return _KEYWORD_QKV
    raise ValueError("FlashAttention-4 provider only supports positional or query/key/value qkv arguments")


def _extract_qkv(args, kwargs):
    """Extract query, key and value from supported SDPA arguments."""
    if len(args) == 3:
        return args
    if len(args) != 0:
        raise ValueError("FlashAttention-4 provider only supports query, key and value positional arguments")

    try:
        return kwargs.pop("query"), kwargs.pop("key"), kwargs.pop("value")
    except KeyError as error:
        raise ValueError("FlashAttention-4 provider requires query, key and value") from error


def _validate_qkv(q, k, v, *, enable_gqa: bool) -> bool:
    """Validate the subset of PyTorch SDPA inputs handled by this provider."""
    if not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
        raise ValueError("FlashAttention-4 provider requires tensor query, key and value")
    if not (q.ndim == k.ndim == v.ndim == 4):
        raise ValueError("FlashAttention-4 provider requires 4D query, key and value")
    if q.shape[0] != k.shape[0] or k.shape[0] != v.shape[0]:
        raise ValueError("FlashAttention-4 provider requires matching batch dimensions")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("FlashAttention-4 provider requires matching key/value sequence lengths")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("FlashAttention-4 provider requires matching query/key head dimensions")
    if k.shape[-3] != v.shape[-3]:
        raise ValueError("FlashAttention-4 provider requires matching key/value head counts")
    if q.shape[-3] == k.shape[-3]:
        return False
    if q.shape[-3] < k.shape[-3] or q.shape[-3] % k.shape[-3] != 0 or not enable_gqa:
        raise ValueError("FlashAttention-4 provider requires valid grouped-query attention")
    return True


def _call_flash_attention4(flash_attn_func, q, k, v, flash_kwargs):
    """Call FlashAttention-4 after copying HND inputs to NHD layout."""
    q_nhd, k_nhd, v_nhd = _hnd_to_nhd(q), _hnd_to_nhd(k), _hnd_to_nhd(v)
    output = flash_attn_func(q_nhd, k_nhd, v_nhd, **flash_kwargs)
    if isinstance(output, tuple):
        output = output[0]
    return _nhd_to_hnd(output)


def _call_flash_attention4_copy_free(flash_attn_func, q, k, v, flash_kwargs):
    """Call FlashAttention-4 when HND inputs are NHD-contiguous views."""
    q_nhd, k_nhd, v_nhd = _hnd_to_nhd_view(q), _hnd_to_nhd_view(k), _hnd_to_nhd_view(v)
    output = flash_attn_func(q_nhd, k_nhd, v_nhd, **flash_kwargs)
    if isinstance(output, tuple):
        output = output[0]
    return _nhd_to_hnd(output)


def _hnd_to_nhd(tensor: torch.Tensor) -> torch.Tensor:
    """Map PyTorch SDPA HND layout to contiguous FlashAttention NHD layout."""
    return tensor.transpose(1, 2).contiguous()


def _hnd_to_nhd_view(tensor: torch.Tensor) -> torch.Tensor:
    """Map HND to NHD without forcing contiguous storage."""
    return tensor.transpose(1, 2)


def _nhd_to_hnd(tensor: torch.Tensor) -> torch.Tensor:
    """Map FlashAttention NHD output back to PyTorch SDPA HND layout."""
    return tensor.transpose(1, 2)


_CALL_IMPLS = {
    False: _call_flash_attention4,
    True: _call_flash_attention4_copy_free,
}
