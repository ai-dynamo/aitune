# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Diffusers attention dispatcher kernel provider."""

from contextlib import nullcontext
from enum import Enum
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

import aitune.torch.backend.kernels.kernel_provider.diffusers_attention_provider as diffusers_provider_module
from aitune.torch.backend.kernels.kernel_optimization_plan import KernelOptimizationPlan
from aitune.torch.backend.kernels.kernel_provider import (
    DiffusersAttentionBackend,
    DiffusersAttentionKernelProvider,
    KernelProviderState,
)

TORCH_MAJOR_MINOR = tuple(int(component) for component in torch.__version__.split(".")[:2])


class _AttentionBackendName(str, Enum):
    FLASH = "flash"


class _FakeAttentionDispatcher:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        *,
        backend=None,
    ):
        self.calls.append((
            query,
            key,
            value,
            {
                "attn_mask": attn_mask,
                "dropout_p": dropout_p,
                "is_causal": is_causal,
                "scale": scale,
                "enable_gqa": enable_gqa,
                "backend": backend,
            },
        ))
        return query + key + value


@pytest.fixture(autouse=True)
def clear_import_cache():
    diffusers_provider_module._import_diffusers_attention_dispatch.cache_clear()
    yield
    diffusers_provider_module._import_diffusers_attention_dispatch.cache_clear()


def _install_runtime(monkeypatch, dispatcher):
    runtime = SimpleNamespace(
        AttentionBackendName=_AttentionBackendName,
        attention_backend=lambda _backend: nullcontext(),
        dispatch_attention_fn=dispatcher,
    )
    monkeypatch.setattr(diffusers_provider_module, "import_module", lambda _module_name: runtime)


def test_provider_name_and_supported_function():
    provider = DiffusersAttentionKernelProvider(DiffusersAttentionBackend.FLASH)

    assert provider.supported_function == "scaled_dot_product_attention"
    assert provider.name == "Diffusers attention flash"
    assert repr(provider) == provider.name


def test_diffusers_attention_backend_values():
    assert {backend.value for backend in DiffusersAttentionBackend.__members__.values()} == {
        "flex",
        "flash",
        "flash_hub",
        "_flash_3",
        "_flash_3_hub",
        "flash_4_hub",
        "sage",
        "sage_hub",
        "_sage_qk_int8_pv_fp8_cuda",
        "_sage_qk_int8_pv_fp8_cuda_sm90",
        "_sage_qk_int8_pv_fp16_cuda",
        "_sage_qk_int8_pv_fp16_triton",
        "xformers",
    }


@pytest.mark.parametrize("backend", [None, 1, "flash", object()])
def test_provider_rejects_non_enum_backend(backend):
    with pytest.raises(TypeError, match="backend must be a DiffusersAttentionBackend"):
        DiffusersAttentionKernelProvider(backend)


@pytest.mark.parametrize(
    "runtime",
    [
        None,
        SimpleNamespace(),
        SimpleNamespace(AttentionBackendName=_AttentionBackendName),
        SimpleNamespace(dispatch_attention_fn=lambda: None),
        SimpleNamespace(
            AttentionBackendName=_AttentionBackendName,
            dispatch_attention_fn=lambda: None,
        ),
    ],
)
def test_provider_rejects_unavailable_dispatcher(monkeypatch, runtime):
    def import_runtime(_module_name):
        if runtime is None:
            raise ImportError
        return runtime

    monkeypatch.setattr(diffusers_provider_module, "import_module", import_runtime)
    provider = DiffusersAttentionKernelProvider(DiffusersAttentionBackend.FLASH)

    with pytest.raises(RuntimeError, match="diffusers>=0.35.0"):
        _ = provider._backend


def test_provider_rejects_backend_unavailable_in_installed_version(monkeypatch):
    _install_runtime(monkeypatch, _FakeAttentionDispatcher())
    provider = DiffusersAttentionKernelProvider(DiffusersAttentionBackend.FLASH_4_HUB)
    query = torch.randn(2, 4, 8, 16)

    assert provider.prepare([((query, query, query), {})]) is True
    with pytest.raises(RuntimeError, match="'flash_4_hub' is not available in the installed version"):
        provider._load_runtime_dependencies()


def test_backend_returns_and_caches_the_bound_dispatcher(monkeypatch):
    dispatcher = _FakeAttentionDispatcher()
    imports = []
    initialized_backends = []

    def attention_backend(backend):
        initialized_backends.append(backend)
        return nullcontext()

    def import_runtime(module_name):
        imports.append(module_name)
        return SimpleNamespace(
            AttentionBackendName=_AttentionBackendName,
            attention_backend=attention_backend,
            dispatch_attention_fn=dispatcher,
        )

    monkeypatch.setattr(diffusers_provider_module, "import_module", import_runtime)
    provider = DiffusersAttentionKernelProvider(DiffusersAttentionBackend.FLASH)

    runtime_function = provider._backend

    assert isinstance(runtime_function, partial)
    assert runtime_function.func is dispatcher
    assert runtime_function.keywords == {"backend": _AttentionBackendName.FLASH}
    assert provider._backend is runtime_function
    assert imports == ["diffusers.models.attention_dispatch"]
    assert initialized_backends == [_AttentionBackendName.FLASH]


def test_load_runtime_dependencies_caches_backend_before_inference(monkeypatch):
    dispatcher = _FakeAttentionDispatcher()
    imports = []

    def import_runtime(module_name):
        imports.append(module_name)
        return SimpleNamespace(
            AttentionBackendName=_AttentionBackendName,
            attention_backend=lambda _backend: nullcontext(),
            dispatch_attention_fn=dispatcher,
        )

    monkeypatch.setattr(diffusers_provider_module, "import_module", import_runtime)
    provider = DiffusersAttentionKernelProvider(DiffusersAttentionBackend.FLASH)

    provider._load_runtime_dependencies()
    provider._load_runtime_dependencies()

    runtime_function = provider._backend
    assert isinstance(runtime_function, partial)
    assert runtime_function.func is dispatcher
    assert imports == ["diffusers.models.attention_dispatch"]


@pytest.mark.parametrize(
    "strict",
    (
        pytest.param(False, id="non-strict"),
        pytest.param(
            True,
            id="strict",
            marks=pytest.mark.skipif(
                TORCH_MAJOR_MINOR == (2, 9),
                reason="PyTorch 2.9 strict export cannot capture hooks that patch torch.nn.functional",
            ),
        ),
    ),
)
def test_runtime_loads_diffusers_before_export(monkeypatch, strict):
    class AttentionModule(torch.nn.Module):
        def forward(self, query, key, value):
            return F.scaled_dot_product_attention(query, key, value)

    def dispatch_attention_fn(query, key, value, *_args, backend=None, **_kwargs):
        assert backend is _AttentionBackendName.FLASH
        return query + key + value

    imports = []

    def import_runtime(module_name):
        imports.append(module_name)
        return SimpleNamespace(
            AttentionBackendName=_AttentionBackendName,
            attention_backend=lambda _backend: nullcontext(),
            dispatch_attention_fn=dispatch_attention_fn,
        )

    monkeypatch.setattr(diffusers_provider_module, "import_module", import_runtime)
    provider = DiffusersAttentionKernelProvider.from_dict({
        "type": "DiffusersAttentionKernelProvider",
        "backend": "flash",
    })
    module = AttentionModule()
    plan = KernelOptimizationPlan((provider,))
    query = torch.randn(1, 1, 4, 8)

    with plan.apply(module):
        imports_before_export = list(imports)
        exported = torch.export.export(module, (query, query, query), strict=strict)

    assert imports_before_export == ["diffusers.models.attention_dispatch"]
    assert imports == imports_before_export
    torch.testing.assert_close(exported.module()(query, query, query), query + query + query)


def test_prepare_and_infer_convert_between_sdpa_and_diffusers_layouts(monkeypatch):
    dispatcher = _FakeAttentionDispatcher()
    _install_runtime(monkeypatch, dispatcher)
    provider = DiffusersAttentionKernelProvider(DiffusersAttentionBackend.FLASH)
    query = torch.randn(2, 4, 8, 16)
    key = torch.randn(2, 4, 8, 16)
    value = torch.randn(2, 4, 8, 16)
    sample = ((query, key, value), {"is_causal": True, "scale": 0.125})

    assert provider.prepare([sample]) is True
    assert provider.state is KernelProviderState.READY
    output = provider(*sample[0], **sample[1])

    called_query, called_key, called_value, kwargs = dispatcher.calls[-1]
    assert called_query.shape == (2, 8, 4, 16)
    assert called_key.shape == (2, 8, 4, 16)
    assert called_value.shape == (2, 8, 4, 16)
    assert kwargs == {
        "attn_mask": None,
        "dropout_p": 0.0,
        "is_causal": True,
        "scale": 0.125,
        "enable_gqa": False,
        "backend": _AttentionBackendName.FLASH,
    }
    torch.testing.assert_close(output, query + key + value)


def test_provider_forwards_keyword_qkv_and_optional_positional_arguments(monkeypatch):
    dispatcher = _FakeAttentionDispatcher()
    _install_runtime(monkeypatch, dispatcher)
    provider = DiffusersAttentionKernelProvider(DiffusersAttentionBackend.FLASH)
    query = torch.randn(2, 4, 8, 16)
    provider.state = KernelProviderState.READY

    provider(query=query, key=query, value=query, attn_mask=None, dropout_p=0.0, enable_gqa=True)
    provider(query, query, query, None, 0.0, True)

    assert dispatcher.calls[0][-1] == {
        "attn_mask": None,
        "dropout_p": 0.0,
        "is_causal": False,
        "scale": None,
        "enable_gqa": True,
        "backend": _AttentionBackendName.FLASH,
    }
    assert dispatcher.calls[1][-1] == {
        "attn_mask": None,
        "dropout_p": 0.0,
        "is_causal": True,
        "scale": None,
        "enable_gqa": False,
        "backend": _AttentionBackendName.FLASH,
    }


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        ((), {}),
        ((None,) * 7, {}),
        ((None,) * 3, {"unknown": True}),
        ((None,) * 3, {"query": None}),
    ],
)
def test_provider_rejects_invalid_sdpa_calls(monkeypatch, args, kwargs):
    _install_runtime(monkeypatch, _FakeAttentionDispatcher())
    provider = DiffusersAttentionKernelProvider.from_dict({
        "type": "DiffusersAttentionKernelProvider",
        "backend": "flash",
    })

    with pytest.raises(TypeError):
        provider(*args, **kwargs)


def test_serialization_round_trip_restores_ready_provider(monkeypatch):
    dispatcher = _FakeAttentionDispatcher()
    _install_runtime(monkeypatch, dispatcher)
    provider = DiffusersAttentionKernelProvider(DiffusersAttentionBackend.FLASH)
    query = torch.randn(2, 4, 8, 16)
    assert provider.prepare([((query, query, query), {})]) is True

    state_dict = provider.to_dict()
    restored = DiffusersAttentionKernelProvider.from_dict(state_dict)
    restored_output = restored(query, query, query)

    assert state_dict == {
        "type": "DiffusersAttentionKernelProvider",
        "backend": "flash",
    }
    assert restored.backend is DiffusersAttentionBackend.FLASH
    assert restored.state is KernelProviderState.READY
    torch.testing.assert_close(restored_output, query * 3)
