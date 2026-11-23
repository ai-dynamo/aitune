# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real Transformers StaticCache integration without pretrained models or downloads."""

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import StaticCache, StaticLayer
from transformers.models.llama import LlamaConfig

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

CACHE_LENGTH = 8
NUM_HEADS = 2
HEAD_DIM = 32
DECODE_STEPS = 3
# The projection and cache scaling differ from eager by up to 0.00176 on the CI runner.
TENSORRT_CACHE_TOLERANCE = Tolerance(rtol=3e-3, atol=3e-3)
TORCH_TENSORRT_JIT_XFAIL = pytest.mark.xfail(
    reason="Torch-TensorRT JIT loses consecutive StaticCache updates; related upstream fix: pytorch/TensorRT#4894",
    raises=(AssertionError, CorrectnessTensorShapeError),
    strict=True,
)
TORCH_TENSORRT_AOT_XFAIL = pytest.mark.xfail(
    reason="Torch-TensorRT AOT fails scalar StaticCache cumulative_length execution or serialization (aitune#188)",
    raises=(RuntimeError, IndexError, AssertionError),
    strict=True,
)


class StaticCacheModel(nn.Module):
    """Project token features and update the actual Transformers cache implementation."""

    def __init__(self, device: torch.device, seed: int = 0):
        super().__init__()
        # A real projection exercises quantizing backends and makes checkpoint weight restoration observable.
        with seeded(device, seed):
            self.projection = nn.Linear(HEAD_DIM, HEAD_DIM, bias=False).to(device)

    def forward(self, value: torch.Tensor, cache: StaticCache, cache_position: torch.Tensor) -> dict:
        key = self.projection(value)
        keys, values = cache.update(key, key * 2, 0, {"cache_position": cache_position})
        return {"keys": keys, "values": values, "cache": cache}


@pytest.mark.matrix("aot_tuning.models.transformers-static-cache-{cache_state}")
@with_backends(
    "base",
    marks={"torch-tensorrt-aot": TORCH_TENSORRT_AOT_XFAIL, "torch-tensorrt-jit": TORCH_TENSORRT_JIT_XFAIL},
)
@pytest.mark.parametrize("cache_state", ["lazy", "initialized"])
def test_static_cache_aot(case: BackendCase, device: torch.device, tmp_path: Path, cache_state: str):
    """Tune all cache states, compare consecutive updates, and repeat after checkpoint reload."""
    backend = case.factory()
    tolerance = _cache_tolerance(backend, case.tolerance)
    tuned = ait.Module(StaticCacheModel(device).eval(), "static-cache", strategy=pinned_strategy(backend))

    def pipeline(value: torch.Tensor):
        return _record_updates(tuned, value, cache_state)

    restored = None
    try:
        with torch.no_grad():
            ait.tune(
                pipeline,
                [_input(device)[0]],
                batch_sizes=[1],
                device=device,
                ignore_failing_modules=False,
            )
        assert isinstance(tuned.module, TunedModule)
        _assert_pinned_backends(tuned.module, backend)
        _check_decode_updates(tuned, device, tolerance, cache_state)

        checkpoint = tmp_path / "static_cache.ait"
        ait.save(tuned, checkpoint)
        tuned.deactivate()
        # A different seed ensures loading restores weights as well as the cache's mutation metadata.
        restored = ait.load(StaticCacheModel(device, seed=1).eval(), checkpoint)
        assert isinstance(restored, ait.Module)
        assert isinstance(restored.module, TunedModule)
        _assert_pinned_backends(restored.module, backend)
        _check_decode_updates(restored, device, tolerance, cache_state)
    finally:
        if isinstance(restored, ait.Module):
            restored.deactivate()
        tuned.deactivate()


@pytest.mark.matrix("jit_tuning.models.transformers-static-cache-{cache_state}")
@with_backends(
    "base",
    marks={"torch-tensorrt-jit": TORCH_TENSORRT_JIT_XFAIL},
)
@pytest.mark.parametrize("cache_state", ["lazy", "initialized"])
def test_static_cache_jit(case: BackendCase, device: torch.device, cache_state: str):
    """JIT tuning must handle initialization and subsequent updates with the requested backend."""
    backend = case.factory()
    tolerance = _cache_tolerance(backend, case.tolerance)
    jit_config.mode = JITMode.TUNE_EAGER
    jit_config.device = device
    jit_config.min_samples = DECODE_STEPS
    jit_config.batch_axis_required = False
    jit_config.strategy = pinned_strategy(backend)
    with prepare_for_jit_tuning():
        model = StaticCacheModel(device).eval()

    # Record the lazy-to-initialized transition before tuning; decoding below uses another fresh cache.
    with torch.no_grad():
        _record_updates(model, _input(device), cache_state)

    (head,) = PatchedModule.heads
    assert head._state == ModuleState.TUNED
    tuned = head._wrapper
    assert isinstance(tuned, TunedModule)
    try:
        _assert_pinned_backends(tuned, backend)
        _check_decode_updates(model, device, tolerance, cache_state)
    finally:
        for selected in tuned.backends.values():
            selected.deactivate()


def _make_cache(device: torch.device, cache_state: str) -> StaticCache:
    config = LlamaConfig(
        num_hidden_layers=1,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        hidden_size=NUM_HEADS * HEAD_DIM,
        head_dim=HEAD_DIM,
        max_position_embeddings=CACHE_LENGTH,
    )
    cache = StaticCache(config=config, max_cache_len=CACHE_LENGTH)
    if cache_state == "initialized":
        cache.early_initialization(
            batch_size=1, num_heads=NUM_HEADS, head_dim=HEAD_DIM, dtype=torch.float32, device=device
        )
    return cache


def _input(device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(7)
    return torch.randn(1, NUM_HEADS, 1, HEAD_DIM, device=device, generator=generator)


def _record_updates(model: Callable, value: torch.Tensor, cache_state: str) -> dict:
    cache = _make_cache(value.device, cache_state)
    output = model(value, cache, torch.zeros(1, dtype=torch.long, device=value.device))
    for position in range(1, DECODE_STEPS):
        output = model(value * (position + 1), cache, torch.tensor([position], device=value.device))
    return output


def _cache_tolerance(backend: Backend, default: Tolerance) -> Tolerance:
    if isinstance(backend, (TensorRTBackend, TorchTensorRTAotBackend, TorchTensorRTJitBackend)):
        return TENSORRT_CACHE_TOLERANCE
    return default


def _assert_pinned_backends(tuned: TunedModule, backend: Backend) -> None:
    # cache_position selects separate prefill and decode graphs even when the cache is already initialized.
    assert len(tuned.backends) == 2
    assert {metadata.llm_phase for metadata in tuned.backends} == {"prefill", "decode"}
    for selected in tuned.backends.values():
        assert type(selected) is type(backend), f"Expected {type(backend).__name__}, got {type(selected).__name__}"
        assert selected.key() == backend.key()


def _check_decode_updates(model: Callable, device: torch.device, tolerance: Tolerance, cache_state: str) -> None:
    reference = StaticCacheModel(device).eval()
    cache = _make_cache(device, cache_state)
    reference_cache = _make_cache(device, cache_state)
    value = _input(device)
    with torch.no_grad():
        for position in range(DECODE_STEPS):
            token = value * (position + 1)
            cache_position = torch.tensor([position], device=device)
            expected = reference(token, reference_cache, cache_position)
            actual = model(token, cache, cache_position)
            tolerance.assert_close(actual["keys"], expected["keys"])
            tolerance.assert_close(actual["values"], expected["values"])
            assert actual["cache"] is cache
            layer, reference_layer = cache.layers[0], reference_cache.layers[0]
            assert isinstance(layer, StaticLayer)
            assert isinstance(reference_layer, StaticLayer)
            assert layer.keys is not None
            assert layer.values is not None
            assert reference_layer.keys is not None
            assert reference_layer.values is not None
            tolerance.assert_close(layer.keys, reference_layer.keys)
            tolerance.assert_close(layer.values, reference_layer.values)
            assert layer.is_initialized
            assert layer.device == reference_layer.device
            assert layer.keys.dtype == reference_layer.keys.dtype
            assert layer.values.dtype == reference_layer.values.dtype
            # Transformers 4 and 5 expose different dimension fields; validate those provided by the reference.
            for field in ("dtype", "max_batch_size", "batch_size", "num_heads", "head_dim", "k_head_dim", "v_head_dim"):
                if hasattr(reference_layer, field):
                    assert getattr(layer, field) == getattr(reference_layer, field), field
