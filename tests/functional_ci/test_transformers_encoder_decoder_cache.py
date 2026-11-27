# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real static EncoderDecoderCache integration without pretrained models or downloads."""

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import EncoderDecoderCache, StaticCache, StaticLayer, WhisperConfig

import aitune.torch as ait
from aitune.torch.backend import Backend, TensorRTBackend, TorchTensorRTAotBackend, TorchTensorRTJitBackend
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.config import config as jit_config
from aitune.torch.jit.patched_module import ModuleState, PatchedModule
from aitune.torch.jit.patcher import prepare_for_jit_tuning
from aitune.torch.module.tuned_module import TunedModule
from aitune.torch.task.correctness import CorrectnessTensorShapeError
from tests.functional_ci.backends import BackendCase, with_backends
from tests.functional_ci.lifecycle import Tolerance, pinned_strategy
from tests.functional_ci.models import seeded

NUM_HEADS = 2
HEAD_DIM = 32
DECODER_LENGTH = 8
ENCODER_LENGTH = 4
DECODE_STEPS = 3
ENCODER_DECODER_BACKEND_MARKS = {
    "torch-tensorrt-aot": pytest.mark.xfail(
        reason="Torch-TensorRT AOT fails scalar encoder-decoder cache execution or serialization (aitune#190)",
        raises=(RuntimeError, IndexError, AssertionError),
        strict=True,
    ),
    "torch-tensorrt-jit": pytest.mark.xfail(
        reason="Torch-TensorRT JIT changes scalar cache output shapes or loses cache updates (aitune#190)",
        raises=(CorrectnessTensorShapeError, AssertionError),
        strict=True,
    ),
}
LAZY_CACHE_BACKEND_MARKS = {
    **ENCODER_DECODER_BACKEND_MARKS,
    "inductor-aot": pytest.mark.xfail(
        reason="Inductor AOT restores lazy cache cumulative_length on CPU instead of CUDA (aitune#190)",
        raises=AssertionError,
        strict=True,
    ),
}


class EncoderDecoderCacheModel(nn.Module):
    """Follow Whisper's self-attention updates and one-time cross-attention initialization."""

    def __init__(self, device: torch.device, seed: int = 0):
        super().__init__()
        # A real weight exercises quantizing backends and checkpoint weight restoration.
        with seeded(device, seed):
            self.projection = nn.Linear(HEAD_DIM, HEAD_DIM, bias=False).to(device)

    def forward(self, value, encoder_value, cache: EncoderDecoderCache, cache_position):
        key = self.projection(value)
        keys, values = cache.self_attention_cache.update(key, key * 2, 0, {"cache_position": cache_position})
        if not cache.is_updated.get(0, False):
            encoder_key = self.projection(encoder_value)
            cache.cross_attention_cache.update(
                encoder_key,
                encoder_key * 3,
                0,
                {"cache_position": torch.arange(encoder_value.shape[-2], device=encoder_value.device)},
            )
            cache.is_updated[0] = True
        cross_layer = cache.cross_attention_cache.layers[0]
        assert isinstance(cross_layer, StaticLayer)
        return {
            "keys": keys,
            "values": values,
            "encoder_keys": cross_layer.keys,
            "encoder_values": cross_layer.values,
            "cache": cache,
        }


@pytest.mark.matrix("aot_tuning.models.transformers-encoder-decoder-cache-lazy")
@with_backends("base", marks=LAZY_CACHE_BACKEND_MARKS)
def test_encoder_decoder_cache_aot_lazy(case: BackendCase, device: torch.device, tmp_path: Path, monkeypatch):
    _check_encoder_decoder_cache_aot(case, device, tmp_path, "lazy", monkeypatch)


@pytest.mark.matrix("aot_tuning.models.transformers-encoder-decoder-cache-initialized")
@with_backends("base", marks=ENCODER_DECODER_BACKEND_MARKS)
def test_encoder_decoder_cache_aot_initialized(case: BackendCase, device: torch.device, tmp_path: Path, monkeypatch):
    _check_encoder_decoder_cache_aot(case, device, tmp_path, "initialized", monkeypatch)


def _check_encoder_decoder_cache_aot(
    case: BackendCase, device: torch.device, tmp_path: Path, cache_state: str, monkeypatch
):
    """Nested cache updates, readiness flags, and caller aliases survive tuning and checkpoint reload."""
    monkeypatch.setattr(ait.config, "strict_mode", True)
    monkeypatch.setattr(ait.config, "enable_transformers_integration", True)
    backend = case.factory()
    tolerance = _cache_tolerance(backend, case.tolerance)
    tuned = ait.Module(
        EncoderDecoderCacheModel(device).eval(), "encoder-decoder-cache", strategy=pinned_strategy(backend)
    )

    def pipeline(value: torch.Tensor):
        _record_updates(tuned, value, cache_state)

    restored = None
    try:
        with torch.no_grad():
            ait.tune(pipeline, [_input(device)[0]], batch_sizes=[1], device=device, ignore_failing_modules=False)
        assert isinstance(tuned.module, TunedModule)
        _assert_tuned_graphs(tuned.module, backend)
        _check_cache_updates(tuned, device, tolerance, cache_state)

        checkpoint = tmp_path / "encoder_decoder_cache.ait"
        ait.save(tuned, checkpoint)
        tuned.deactivate()
        # A different seed proves that reload restores weights as well as mutation metadata.
        restored = ait.load(EncoderDecoderCacheModel(device, seed=1).eval(), checkpoint)
        assert isinstance(restored, ait.Module)
        assert isinstance(restored.module, TunedModule)
        _assert_tuned_graphs(restored.module, backend)
        _check_cache_updates(restored, device, tolerance, cache_state)
    finally:
        if isinstance(restored, ait.Module):
            restored.deactivate()
        tuned.deactivate()


@pytest.mark.matrix("jit_tuning.models.transformers-encoder-decoder-cache-lazy")
@with_backends("base", marks=LAZY_CACHE_BACKEND_MARKS)
def test_encoder_decoder_cache_jit_lazy(case: BackendCase, device: torch.device, monkeypatch):
    _check_encoder_decoder_cache_jit(case, device, "lazy", monkeypatch)


@pytest.mark.matrix("jit_tuning.models.transformers-encoder-decoder-cache-initialized")
@with_backends("base", marks=ENCODER_DECODER_BACKEND_MARKS)
def test_encoder_decoder_cache_jit_initialized(case: BackendCase, device: torch.device, monkeypatch):
    _check_encoder_decoder_cache_jit(case, device, "initialized", monkeypatch)


def _check_encoder_decoder_cache_jit(case: BackendCase, device: torch.device, cache_state: str, monkeypatch):
    """JIT tuning separates cross-attention initialization and reuse within the same decode phase."""
    monkeypatch.setattr(ait.config, "strict_mode", True)
    monkeypatch.setattr(ait.config, "enable_transformers_integration", True)
    backend = case.factory()
    tolerance = _cache_tolerance(backend, case.tolerance)
    jit_config.mode = JITMode.TUNE_EAGER
    jit_config.device = device
    jit_config.min_samples = 2 * DECODE_STEPS
    jit_config.batch_axis_required = False
    jit_config.strategy = pinned_strategy(backend)
    with prepare_for_jit_tuning():
        model = EncoderDecoderCacheModel(device).eval()

    with torch.no_grad():
        _record_updates(model, _input(device), cache_state)

    (head,) = PatchedModule.heads
    assert head._state == ModuleState.TUNED
    tuned = head._wrapper
    assert isinstance(tuned, TunedModule)
    try:
        _assert_tuned_graphs(tuned, backend)
        _check_cache_updates(model, device, tolerance, cache_state)
    finally:
        for selected in tuned.backends.values():
            selected.deactivate()


def _make_cache(device: torch.device, cache_state: str) -> EncoderDecoderCache:
    config = WhisperConfig(
        d_model=NUM_HEADS * HEAD_DIM,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=NUM_HEADS,
        decoder_attention_heads=NUM_HEADS,
        max_source_positions=ENCODER_LENGTH,
        max_target_positions=DECODER_LENGTH,
    )
    caches = [StaticCache(config=config, max_cache_len=length) for length in (DECODER_LENGTH, ENCODER_LENGTH)]
    if cache_state == "initialized":
        for nested_cache in caches:
            nested_cache.early_initialization(
                batch_size=1, num_heads=NUM_HEADS, head_dim=HEAD_DIM, dtype=torch.float32, device=device
            )
    cache = EncoderDecoderCache(*caches)
    cache.is_updated[0] = False
    return cache


def _input(device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(7)
    return torch.randn(1, NUM_HEADS, 1, HEAD_DIM, device=device, generator=generator)


def _record_updates(model: Callable, value: torch.Tensor, cache_state: str) -> None:
    # Starting once in decode exercises readiness=False and readiness=True in the same LLM phase.
    for start_position in (0, 1):
        cache = _make_cache(value.device, cache_state)
        for step in range(DECODE_STEPS):
            position = torch.tensor([start_position + step], device=value.device)
            encoder_value = value.expand(-1, -1, ENCODER_LENGTH, -1).contiguous() * (step + 2)
            model(value * (step + 1), encoder_value, cache, position)


def _assert_tuned_graphs(tuned: TunedModule, backend: Backend) -> None:
    readiness_path = ("cache", "is_updated", 0)
    phase_readiness = {
        (metadata.llm_phase, {locator.path: value for locator, value in metadata.other_data}[readiness_path])
        for metadata in tuned.backends
    }
    assert len(tuned.backends) == 3
    assert phase_readiness == {("prefill", False), ("decode", False), ("decode", True)}
    for selected in tuned.backends.values():
        assert type(selected) is type(backend), f"Expected {type(backend).__name__}, got {type(selected).__name__}"
        assert selected.key() == backend.key()


def _cache_tolerance(backend: Backend, default: Tolerance) -> Tolerance:
    if isinstance(backend, TensorRTBackend):
        # Standalone TensorRT differs from eager by up to 0.00528 on the CI runner.
        return Tolerance(rtol=3e-3, atol=6e-3)
    if isinstance(backend, (TorchTensorRTAotBackend, TorchTensorRTJitBackend)):
        return Tolerance(rtol=3e-3, atol=3e-3)
    return default


def _check_cache_updates(model: Callable, device: torch.device, tolerance: Tolerance, cache_state: str) -> None:
    reference = EncoderDecoderCacheModel(device).eval()
    caches = [_make_cache(device, cache_state) for _ in range(2)]
    expected_caches = [_make_cache(device, cache_state) for _ in range(2)]
    value = _input(device)
    with torch.no_grad():
        for step in range(DECODE_STEPS):
            for caller in range(2):
                cache, expected_cache = caches[caller], expected_caches[caller]
                self_cache, cross_cache = cache.self_attention_cache, cache.cross_attention_cache
                token = value * (step + 1) * (caller + 1)
                # Changing encoder inputs must not overwrite cross-attention state after its first update.
                encoder_value = value.expand(-1, -1, ENCODER_LENGTH, -1).contiguous() * (step + 2) * (caller + 1)
                position = torch.tensor([caller + step], device=device)
                expected = reference(token, encoder_value, expected_cache, position)
                actual = model(token, encoder_value, cache, position)
                for field in ("keys", "values", "encoder_keys", "encoder_values"):
                    tolerance.assert_close(actual[field], expected[field])
                assert actual["cache"] is cache
                assert cache.self_attention_cache is self_cache
                assert cache.cross_attention_cache is cross_cache
                self_layer, cross_layer = self_cache.layers[0], cross_cache.layers[0]
                assert isinstance(self_layer, StaticLayer)
                assert isinstance(cross_layer, StaticLayer)
                assert actual["keys"] is self_layer.keys
                assert actual["encoder_keys"] is cross_layer.keys
                assert cache.is_updated[0] is True
                # Updating one caller must also leave the other caller's cache unchanged.
                for current, current_expected in zip(caches, expected_caches, strict=True):
                    _assert_cache_matches(current, current_expected, tolerance)


def _assert_cache_matches(actual: EncoderDecoderCache, expected: EncoderDecoderCache, tolerance: Tolerance) -> None:
    assert actual.is_updated == expected.is_updated
    for name in ("self_attention_cache", "cross_attention_cache"):
        layer, expected_layer = getattr(actual, name).layers[0], getattr(expected, name).layers[0]
        assert isinstance(layer, StaticLayer)
        assert isinstance(expected_layer, StaticLayer)
        assert layer.is_initialized == expected_layer.is_initialized
        for field in ("keys", "values", "cumulative_length"):
            if hasattr(expected_layer, field):
                tolerance.assert_close(getattr(layer, field), getattr(expected_layer, field))
        # Transformers 4 and 5 expose different initialization fields; compare those available.
        for field in (
            "device",
            "dtype",
            "max_batch_size",
            "batch_size",
            "num_heads",
            "head_dim",
            "k_head_dim",
            "v_head_dim",
        ):
            if hasattr(expected_layer, field):
                assert getattr(layer, field) == getattr(expected_layer, field), field
