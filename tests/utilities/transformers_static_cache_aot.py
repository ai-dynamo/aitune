# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared functional validation for AOT backends using Transformers ``StaticCache``."""

import gc
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from transformers import StaticCache
from transformers.models.llama import LlamaConfig

from aitune.torch import Module, OneBackendStrategy, load, save, tune
from aitune.torch.backend.backend import Backend
from aitune.torch.module.wrapper_module import ModuleState
from aitune.torch.module_registry import MODULE_REGISTRY

CACHE_LENGTH = 8
NUM_HEADS = 2
HEAD_DIM = 4


class StaticCacheModule(nn.Module):
    """Update a Transformers cache and return the mutated cache tensors."""

    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        past_key_values: StaticCache,
        cache_position: torch.Tensor,
    ) -> dict:
        """Mutate the cache fields used by Transformers decode calls."""
        keys, values = past_key_values.update(key, value, 0, {"cache_position": cache_position})
        return {"keys": keys, "values": values, "past_key_values": past_key_values}


def _make_cache(device: torch.device, batch_size: int = 1) -> StaticCache:
    config = LlamaConfig(
        num_hidden_layers=1,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        hidden_size=NUM_HEADS * HEAD_DIM,
        head_dim=HEAD_DIM,
        max_position_embeddings=CACHE_LENGTH,
    )
    cache = StaticCache(config=config, max_cache_len=CACHE_LENGTH)
    cache.early_initialization(
        batch_size=batch_size,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        dtype=torch.float32,
        device=device,
    )
    return cache


def _validate_cache_updates(module: nn.Module, device: torch.device) -> None:
    """Run two decode calls and compare their outputs and cache mutations with eager."""
    eager_module = StaticCacheModule().eval().to(device)
    eager_cache = _make_cache(device)
    tuned_cache = _make_cache(device)

    for cache_position, scale in enumerate((1.0, 3.0)):
        key = torch.full((1, NUM_HEADS, 1, HEAD_DIM), scale, device=device)
        value = key * 2
        position = torch.tensor([cache_position], device=device)
        full_cache_copies = []
        runtime_layer = tuned_cache.layers[0]
        original_copy = torch.Tensor.copy_

        def track_copy(
            target,
            source,
            *args,
            _runtime_layer=runtime_layer,
            _full_cache_copies=full_cache_copies,
            _original_copy=original_copy,
            **kwargs,
        ):
            if target is _runtime_layer.keys or target is _runtime_layer.values:
                _full_cache_copies.append((tuple(target.shape), source.numel()))
            return _original_copy(target, source, *args, **kwargs)

        with torch.no_grad():
            expected = eager_module(key, value, eager_cache, position)
            with patch.object(torch.Tensor, "copy_", track_copy):
                actual = module(key, value, tuned_cache, position)

        torch.testing.assert_close(actual["keys"], expected["keys"])
        torch.testing.assert_close(actual["values"], expected["values"])
        torch.testing.assert_close(tuned_cache.layers[0].keys, eager_cache.layers[0].keys)
        torch.testing.assert_close(tuned_cache.layers[0].values, eager_cache.layers[0].values)
        assert actual["past_key_values"] is tuned_cache
        assert full_cache_copies == [], (
            f"A single-position decode update copied complete caller-owned KV-cache tensors: {full_cache_copies}."
        )


def run_static_cache_aot_test(backend: Backend) -> None:
    """Tune, execute, save, and reload an AOT backend that mutates ``StaticCache``."""
    device = torch.device("cuda")
    model = StaticCacheModule().eval().to(device)
    strategy = OneBackendStrategy(backend)
    strategy.enable_find_max_batch_size(False)
    strategy.enable_performance_validation(False)
    tuned_module = Module(model, type(backend).__name__, strategy=strategy)
    record_cache: StaticCache | None = None
    next_cache_position = 0

    def record(value: torch.Tensor) -> object:
        nonlocal next_cache_position, record_cache
        key = value.to(device)
        if record_cache is None:
            record_cache = _make_cache(device, batch_size=key.shape[0])
        sequence_length = int(key.shape[-2])
        cache_position = torch.arange(next_cache_position, next_cache_position + sequence_length, device=device)
        output = tuned_module(key, key * 2, record_cache, cache_position)
        next_cache_position += sequence_length
        return output

    dataset = [
        torch.ones(NUM_HEADS, 1, HEAD_DIM),
        torch.full((NUM_HEADS, 1, HEAD_DIM), 3.0),
    ]
    loaded_module = None
    try:
        tune(
            record,
            dataset,
            batch_sizes=[1],
            max_num_batches_per_batch_size=2,
            device=device,
            ignore_failing_modules=False,
        )
        assert tuned_module.state == ModuleState.TUNED
        _validate_cache_updates(tuned_module, device)

        with tempfile.TemporaryDirectory(prefix="aitune-static-cache-") as checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir) / "model.ait"
            save(tuned_module, checkpoint_path)
            loaded_module = load(StaticCacheModule().eval().to(device), checkpoint_path)
            assert isinstance(loaded_module, Module)
            assert loaded_module.state == ModuleState.TUNED
            _validate_cache_updates(loaded_module, device)
    finally:
        tuned_module.deactivate()
        MODULE_REGISTRY.clear()
        record_cache = None
        del loaded_module, model, dataset
        gc.collect()
        torch.cuda.empty_cache()
