# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the FlashAttention-4 kernel provider."""

import logging
from types import MethodType, SimpleNamespace

import pytest
import torch

import aitune.torch.backend.kernels.kernel_provider.flash_attention4_provider as flash_attention4_provider_module
from aitune.torch.backend.kernels.kernel_provider import FlashAttention4KernelProvider, KernelProviderState


class _FakeFlashAttention4Backend:
    def __init__(self):
        self.calls = []

    def flash_attn_func(self, q, k, v, **kwargs):
        self.calls.append((q, k, v, kwargs))
        return torch.empty_like(q), None


def _qkv(*, copy_free_layout: bool, query_heads: int = 4, key_value_heads: int = 4):
    if copy_free_layout:
        query = torch.randn(2, 17, query_heads, 8).transpose(1, 2)
        key = torch.randn(2, 19, key_value_heads, 8).transpose(1, 2)
        value = torch.randn(2, 19, key_value_heads, 8).transpose(1, 2)
    else:
        query = torch.randn(2, query_heads, 17, 8)
        key = torch.randn(2, key_value_heads, 19, 8)
        value = torch.randn(2, key_value_heads, 19, 8)
    return query, key, value


def test_flash_attention4_provider_name_and_supported_function(monkeypatch):
    monkeypatch.setattr(flash_attention4_provider_module, "_flash_attention4_version", lambda: "v4.0.0b27")
    provider = FlashAttention4KernelProvider()

    assert provider.supported_function == "scaled_dot_product_attention"
    assert provider.name == "FlashAttention-4 v4.0.0b27"
    assert repr(provider) == provider.name


def test_flash_attention4_provider_name_when_runtime_cannot_be_imported(monkeypatch):
    def missing_runtime(_module_name):
        raise ImportError

    version = flash_attention4_provider_module._flash_attention4_version
    version.cache_clear()
    monkeypatch.setattr(flash_attention4_provider_module, "import_module", missing_runtime)

    try:
        assert FlashAttention4KernelProvider().name == "FlashAttention-4 cannot be imported"
    finally:
        version.cache_clear()


def test_flash_attention4_provider_name_uses_runtime_version(monkeypatch):
    version = flash_attention4_provider_module._flash_attention4_version
    version.cache_clear()
    monkeypatch.setattr(
        flash_attention4_provider_module,
        "import_module",
        lambda _module_name: SimpleNamespace(__version__="4.0.0b27"),
    )

    try:
        assert FlashAttention4KernelProvider().name == "FlashAttention-4 v4.0.0b27"
    finally:
        version.cache_clear()


def test_flash_attention4_provider_name_uses_fallback_package_version(monkeypatch):
    requested_packages = []

    def package_version(package_name):
        requested_packages.append(package_name)
        if package_name == "flash-attn-4":
            raise flash_attention4_provider_module.PackageNotFoundError
        return "4.0.0b26"

    version = flash_attention4_provider_module._flash_attention4_version
    version.cache_clear()
    monkeypatch.setattr(
        flash_attention4_provider_module,
        "import_module",
        lambda _module_name: SimpleNamespace(__version__="0.0.0"),
    )
    monkeypatch.setattr(flash_attention4_provider_module, "pkg_version", package_version)

    try:
        assert FlashAttention4KernelProvider().name == "FlashAttention-4 v4.0.0b26"
        assert requested_packages == ["flash-attn-4", "fa4"]
    finally:
        version.cache_clear()


def test_flash_attention4_provider_name_when_package_version_is_unknown(monkeypatch):
    def missing_package(_package_name):
        raise flash_attention4_provider_module.PackageNotFoundError

    version = flash_attention4_provider_module._flash_attention4_version
    version.cache_clear()
    monkeypatch.setattr(flash_attention4_provider_module, "import_module", lambda _module_name: SimpleNamespace())
    monkeypatch.setattr(flash_attention4_provider_module, "pkg_version", missing_package)

    try:
        assert FlashAttention4KernelProvider().name == "FlashAttention-4 unknown version"
    finally:
        version.cache_clear()


@pytest.mark.parametrize("backend", [None, SimpleNamespace(), SimpleNamespace(flash_attn_func=None)])
def test_backend_rejects_an_unavailable_runtime(monkeypatch, backend):
    def import_backend(_module_name):
        if backend is None:
            raise ImportError
        return backend

    monkeypatch.setattr(flash_attention4_provider_module, "import_module", import_backend)
    provider = FlashAttention4KernelProvider()

    with pytest.raises(RuntimeError, match="pip install --pre flash-attn-4"):
        _ = provider._backend


def test_backend_returns_and_caches_the_runtime_function(monkeypatch):
    backend = _FakeFlashAttention4Backend()
    imports = []

    def import_backend(module_name):
        imports.append(module_name)
        return backend

    monkeypatch.setattr(flash_attention4_provider_module, "import_module", import_backend)
    provider = FlashAttention4KernelProvider()

    runtime_function = provider._backend

    assert callable(runtime_function)
    assert isinstance(runtime_function, MethodType)
    assert runtime_function.__self__ is backend
    assert provider._backend is runtime_function
    assert imports == ["flash_attn.cute"]


def test_prepare_does_not_load_or_run_the_runtime(monkeypatch):
    def unexpected_import(_module_name):
        raise AssertionError("prepare must not import FlashAttention-4")

    monkeypatch.setattr(flash_attention4_provider_module, "import_module", unexpected_import)
    provider = FlashAttention4KernelProvider()
    query, key, value = _qkv(copy_free_layout=False)

    assert provider.prepare([((query, key, value), {})]) is True
    assert provider.state is KernelProviderState.READY


def test_prepare_rejects_empty_unsupported_or_inconsistent_samples():
    provider = FlashAttention4KernelProvider()
    query, key, value = _qkv(copy_free_layout=False)
    positional_sample = ((query, key, value), {})
    keyword_sample = ((), {"query": query, "key": key, "value": value})

    assert provider.prepare([]) is False
    assert provider.prepare([((query, key, value), {"attn_mask": torch.ones(17, 19)})]) is False
    assert provider.prepare([positional_sample, keyword_sample]) is False


@pytest.mark.parametrize(
    ("unsupported_kwargs", "message"),
    [
        ({"dropout_p": 0.1}, "dropout_p=0.0"),
        ({"unsupported": True}, "does not support kwargs"),
    ],
)
def test_prepare_rejects_unsupported_sdpa_kwargs(unsupported_kwargs, message, caplog):
    query, key, value = _qkv(copy_free_layout=False)

    with caplog.at_level(logging.DEBUG, logger=flash_attention4_provider_module.__name__):
        assert FlashAttention4KernelProvider().prepare([((query, key, value), unsupported_kwargs)]) is False

    assert message in caplog.text


@pytest.mark.parametrize(
    "invalid_case",
    [
        "non_tensor",
        "rank",
        "batch",
        "key_value_sequence",
        "head_dimension",
        "key_value_heads",
        "grouped_query_attention",
    ],
)
def test_prepare_rejects_invalid_qkv(invalid_case):
    query, key, value = _qkv(copy_free_layout=False)
    if invalid_case == "non_tensor":
        query = None
    elif invalid_case == "rank":
        query = query[0]
    elif invalid_case == "batch":
        key = torch.randn(3, 4, 19, 8)
    elif invalid_case == "key_value_sequence":
        value = torch.randn(2, 4, 20, 8)
    elif invalid_case == "head_dimension":
        key = torch.randn(2, 4, 19, 7)
    elif invalid_case == "key_value_heads":
        value = torch.randn(2, 2, 19, 8)
    elif invalid_case == "grouped_query_attention":
        query, key, value = _qkv(copy_free_layout=False, query_heads=4, key_value_heads=2)

    assert FlashAttention4KernelProvider().prepare([((query, key, value), {})]) is False


def test_prepare_rejects_incomplete_positional_qkv():
    query, key, _value = _qkv(copy_free_layout=False)

    assert FlashAttention4KernelProvider().prepare([((query, key), {})]) is False


@pytest.mark.parametrize("keyword_qkv", [False, True])
@pytest.mark.parametrize("copy_free_layout", [False, True])
def test_prepare_selects_and_runs_the_sample_specific_plan(keyword_qkv, copy_free_layout):
    backend = _FakeFlashAttention4Backend()
    provider = FlashAttention4KernelProvider()
    provider.__dict__["_backend"] = backend.flash_attn_func
    query, key, value = _qkv(copy_free_layout=copy_free_layout)
    args = () if keyword_qkv else (query, key, value)
    kwargs = (
        {"query": query, "key": key, "value": value, "is_causal": True, "scale": 0.125}
        if keyword_qkv
        else {"is_causal": True, "scale": 0.125}
    )

    assert provider.prepare([(args, kwargs)]) is True
    assert provider.state is KernelProviderState.READY
    assert provider.qkv_source == (
        flash_attention4_provider_module._KEYWORD_QKV
        if keyword_qkv
        else flash_attention4_provider_module._POSITIONAL_QKV
    )
    assert provider.copy_free_layout is copy_free_layout
    assert provider.flash_kwargs == {"causal": True, "softmax_scale": 0.125}
    assert backend.calls == []

    output = provider(*args, **kwargs)

    called_query, called_key, called_value, flash_kwargs = backend.calls[0]
    assert output.shape == query.shape
    assert called_query.shape == (2, 17, 4, 8)
    assert called_key.shape == (2, 19, 4, 8)
    assert called_value.shape == (2, 19, 4, 8)
    assert flash_kwargs == {"causal": True, "softmax_scale": 0.125}
    if copy_free_layout:
        assert called_query.data_ptr() == query.transpose(1, 2).data_ptr()
        assert called_key.data_ptr() == key.transpose(1, 2).data_ptr()
        assert called_value.data_ptr() == value.transpose(1, 2).data_ptr()


def test_prepare_uses_safe_non_packed_gqa_without_probing_runtime():
    backend = _FakeFlashAttention4Backend()
    provider = FlashAttention4KernelProvider()
    provider.__dict__["_backend"] = backend.flash_attn_func
    query, key, value = _qkv(copy_free_layout=True, query_heads=4, key_value_heads=2)

    assert provider.prepare([((query, key, value), {"enable_gqa": True})]) is True
    assert provider.flash_kwargs == {"causal": False, "pack_gqa": False}
    assert backend.calls == []

    provider(query, key, value)
    assert backend.calls[0][3] == {"causal": False, "pack_gqa": False}


def test_serialization_round_trip_restores_ready_provider(monkeypatch):
    backend = _FakeFlashAttention4Backend()
    provider = FlashAttention4KernelProvider()
    query, key, value = _qkv(copy_free_layout=True)
    sample = ((), {"query": query, "key": key, "value": value, "is_causal": True})
    assert provider.prepare([sample]) is True

    state_dict = provider.to_dict()
    restored = FlashAttention4KernelProvider.from_dict(state_dict)
    monkeypatch.setattr(flash_attention4_provider_module, "import_module", lambda _module_name: backend)

    assert state_dict == {
        "type": "FlashAttention4KernelProvider",
        "qkv_source": "keyword_qkv",
        "flash_kwargs": {"causal": True},
        "copy_free_layout": True,
    }
    assert restored.state is KernelProviderState.READY
    output = restored(query=query, key=key, value=value)
    assert output.shape == query.shape


def test_restored_provider_reports_missing_runtime(monkeypatch):
    monkeypatch.setattr(flash_attention4_provider_module, "import_module", lambda _module_name: SimpleNamespace())
    restored = FlashAttention4KernelProvider.from_dict({
        "type": "FlashAttention4KernelProvider",
        "qkv_source": "positional_qkv",
        "flash_kwargs": {"causal": False},
        "copy_free_layout": False,
    })
    query, key, value = _qkv(copy_free_layout=False)

    with pytest.raises(RuntimeError, match="pip install --pre flash-attn-4"):
        restored(query, key, value)
