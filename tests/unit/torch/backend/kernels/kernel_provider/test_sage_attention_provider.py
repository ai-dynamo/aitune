# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the SageAttention kernel provider."""

from types import MethodType, SimpleNamespace

import pytest
import torch

import aitune.torch.backend.kernels.kernel_provider.sage_attention_provider as sage_attention_provider_module
from aitune.torch.backend.kernels.kernel_provider import KernelProviderState, SageAttentionKernelProvider


class _FakeSageAttentionBackend:
    def __init__(self):
        self.calls = []

    def sageattn(self, q, k, v, tensor_layout="HND", **kwargs):
        self.calls.append((q, k, v, tensor_layout, kwargs))
        return torch.empty_like(q).contiguous()


def _hnd_view():
    return torch.randn(2, 17, 4, 8).transpose(1, 2)


def test_sage_attention_provider_name_and_supported_function(monkeypatch):
    monkeypatch.setattr(sage_attention_provider_module, "_sageattention_version", lambda: "v2.2.0")
    provider = SageAttentionKernelProvider()

    assert provider.supported_function == "scaled_dot_product_attention"
    assert provider.name == "Sage Attention v2.2.0"
    assert repr(provider) == provider.name


def test_sage_attention_provider_name_when_runtime_cannot_be_imported(monkeypatch):
    def missing_runtime(_module_name):
        raise ImportError

    version = sage_attention_provider_module._sageattention_version
    version.cache_clear()
    monkeypatch.setattr(sage_attention_provider_module, "import_module", missing_runtime)

    try:
        assert SageAttentionKernelProvider().name == "Sage Attention cannot be imported"
    finally:
        version.cache_clear()


def test_sage_attention_provider_name_uses_runtime_version(monkeypatch):
    version = sage_attention_provider_module._sageattention_version
    version.cache_clear()
    monkeypatch.setattr(
        sage_attention_provider_module,
        "import_module",
        lambda _module_name: SimpleNamespace(__version__="2.2.0"),
    )

    try:
        assert SageAttentionKernelProvider().name == "Sage Attention v2.2.0"
    finally:
        version.cache_clear()


def test_sage_attention_provider_name_uses_package_version(monkeypatch):
    version = sage_attention_provider_module._sageattention_version
    version.cache_clear()
    monkeypatch.setattr(sage_attention_provider_module, "import_module", lambda _module_name: SimpleNamespace())
    monkeypatch.setattr(sage_attention_provider_module, "pkg_version", lambda _package_name: "2.1.1")

    try:
        assert SageAttentionKernelProvider().name == "Sage Attention v2.1.1"
    finally:
        version.cache_clear()


def test_sage_attention_provider_name_when_package_version_is_unknown(monkeypatch):
    def missing_package(_package_name):
        raise sage_attention_provider_module.PackageNotFoundError

    version = sage_attention_provider_module._sageattention_version
    version.cache_clear()
    monkeypatch.setattr(sage_attention_provider_module, "import_module", lambda _module_name: SimpleNamespace())
    monkeypatch.setattr(sage_attention_provider_module, "pkg_version", missing_package)

    try:
        assert SageAttentionKernelProvider().name == "Sage Attention unknown version"
    finally:
        version.cache_clear()


@pytest.mark.parametrize("backend", [None, SimpleNamespace(), SimpleNamespace(sageattn=None)])
def test_backend_rejects_an_unavailable_runtime(monkeypatch, backend):
    def import_backend(_module_name):
        if backend is None:
            raise ImportError
        return backend

    monkeypatch.setattr(sage_attention_provider_module, "import_module", import_backend)
    provider = SageAttentionKernelProvider()

    with pytest.raises(RuntimeError, match="pip install sageattention"):
        _ = provider._backend


def test_prepare_does_not_load_the_runtime(monkeypatch):
    def unexpected_import(_module_name):
        raise AssertionError("prepare must not import SageAttention")

    monkeypatch.setattr(sage_attention_provider_module, "import_module", unexpected_import)
    provider = SageAttentionKernelProvider()
    query = torch.randn(2, 4, 17, 8)

    assert provider.prepare([((query, query, query), {})]) is True
    assert provider.state is KernelProviderState.READY


def test_backend_returns_and_caches_the_runtime_function(monkeypatch):
    backend = _FakeSageAttentionBackend()
    imports = []

    def import_backend(module_name):
        imports.append(module_name)
        return backend

    monkeypatch.setattr(sage_attention_provider_module, "import_module", import_backend)
    provider = SageAttentionKernelProvider()

    runtime_function = provider._backend

    assert callable(runtime_function)
    assert isinstance(runtime_function, MethodType)
    assert runtime_function.__self__ is backend
    assert provider._backend is runtime_function
    assert imports == ["sageattention"]


def test_prepare_rejects_empty_or_inconsistent_samples():
    provider = SageAttentionKernelProvider()
    query = torch.randn(2, 4, 17, 8)
    positional_sample = ((query, query, query), {})
    mapped_sample = ((), {"query": query, "key": query, "value": query})

    assert provider.prepare([]) is False
    assert provider.prepare([positional_sample, mapped_sample]) is False


def test_prepare_does_not_run_inference():
    def failing_sageattn(*_args, **_kwargs):
        raise RuntimeError("unsupported sample")

    provider = SageAttentionKernelProvider()
    provider.__dict__["_backend"] = failing_sageattn
    query = _hnd_view()
    sample = ((), {"query": query, "key": query, "value": query})

    assert provider.prepare([sample]) is True
    assert provider.state is KernelProviderState.READY

    with pytest.raises(RuntimeError, match="unsupported sample"):
        provider(**sample[1])


@pytest.mark.parametrize(
    ("needs_kwargs_mapping", "use_diffusers_native_hnd_view", "expected_layout"),
    [
        (False, False, "HND"),
        (True, False, "HND"),
        (False, True, "NHD"),
        (True, True, "NHD"),
    ],
)
def test_prepare_selects_and_runs_the_sample_specific_plan(
    needs_kwargs_mapping,
    use_diffusers_native_hnd_view,
    expected_layout,
):
    backend = _FakeSageAttentionBackend()
    provider = SageAttentionKernelProvider()
    provider.__dict__["_backend"] = backend.sageattn
    query = _hnd_view() if use_diffusers_native_hnd_view else torch.randn(2, 4, 17, 8)
    key = _hnd_view() if use_diffusers_native_hnd_view else torch.randn(2, 4, 17, 8)
    value = _hnd_view() if use_diffusers_native_hnd_view else torch.randn(2, 4, 17, 8)
    kwargs = (
        {"query": query, "key": key, "value": value} if needs_kwargs_mapping else {"q": query, "k": key, "v": value}
    )

    assert provider.prepare([((), kwargs)]) is True
    assert provider.state is KernelProviderState.READY
    assert provider.needs_kwargs_mapping is needs_kwargs_mapping
    assert provider.use_diffusers_native_hnd_view is use_diffusers_native_hnd_view
    assert backend.calls == []

    output = provider(**kwargs, scale=0.125)

    called_query, called_key, called_value, tensor_layout, sage_kwargs = backend.calls[0]
    expected_shape = (2, 17, 4, 8) if use_diffusers_native_hnd_view else (2, 4, 17, 8)
    assert output.shape == query.shape
    assert called_query.shape == expected_shape
    assert called_key.shape == expected_shape
    assert called_value.shape == expected_shape
    assert tensor_layout == expected_layout
    assert sage_kwargs == {"sm_scale": 0.125}


def test_serialization_round_trip_restores_ready_provider(monkeypatch):
    backend = _FakeSageAttentionBackend()
    provider = SageAttentionKernelProvider()
    provider.__dict__["_backend"] = backend.sageattn
    query = _hnd_view()
    sample = ((), {"query": query, "key": query, "value": query})
    assert provider.prepare([sample]) is True

    state_dict = provider.to_dict()
    restored = SageAttentionKernelProvider.from_dict(state_dict)
    monkeypatch.setattr(sage_attention_provider_module, "import_module", lambda _module_name: backend)

    assert state_dict == {
        "type": "SageAttentionKernelProvider",
        "needs_kwargs_mapping": True,
        "use_diffusers_native_hnd_view": True,
    }
    assert restored.state is KernelProviderState.READY
    output = restored(query=query, key=query, value=query)
    assert output.shape == query.shape


def test_restored_provider_reports_missing_runtime(monkeypatch):
    monkeypatch.setattr(sage_attention_provider_module, "import_module", lambda _module_name: SimpleNamespace())
    restored = SageAttentionKernelProvider.from_dict({
        "type": "SageAttentionKernelProvider",
        "needs_kwargs_mapping": False,
        "use_diffusers_native_hnd_view": False,
    })
    query = torch.randn(2, 4, 17, 8)

    with pytest.raises(RuntimeError, match="pip install sageattention"):
        restored(query, query, query)
