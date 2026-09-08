# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Transformers cache integration."""

import pytest
import torch
import torch.nn as nn

from aitune.torch.libs.exporting import TorchExporter
from tests.utilities.helpers import make_graph_spec, make_recorded_graph_spec, update_input_spec

transformers = pytest.importorskip("transformers")
from transformers import StaticCache  # noqa: E402
from transformers.models.llama import LlamaConfig  # noqa: E402


def _make_cache() -> StaticCache:
    config = LlamaConfig(
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=8,
        head_dim=4,
        max_position_embeddings=8,
    )
    return StaticCache(config=config, max_cache_len=8)


def _make_initialized_cache() -> StaticCache:
    cache = _make_cache()
    cache.early_initialization(
        batch_size=1,
        num_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    return cache


class _UpdateCache(nn.Module):
    def forward(self, value, cache):
        cache.layers[0].keys.add_(value)
        return {"value": cache.layers[0].keys[..., :1, :], "cache": cache}


class _InitializeCache(nn.Module):
    def forward(self, key, cache, cache_position):
        keys, _ = cache.update(key, key, 0, {"cache_position": cache_position})
        return keys


class _ReadCache(nn.Module):
    def forward(self, value, cache):
        layer = cache.layers[0]
        return value + layer.keys.sum() + layer.values.sum()


def test_initialized_static_cache_preserves_caller_owned_output_identity():
    module = _UpdateCache().eval()
    recorded_cache = _make_initialized_cache()
    value = torch.ones_like(recorded_cache.layers[0].keys)
    sample = ((value, recorded_cache), {})
    graph_spec = make_recorded_graph_spec(module, sample)
    exported = TorchExporter().export(module, sample, graph_spec).exported_program.module()

    runtime_cache = _make_initialized_cache()
    arguments = graph_spec.forward_signature.normalize((value, runtime_cache), {}).arguments
    flat_inputs = graph_spec.input_spec.flatten_tensors(arguments)
    output = graph_spec.restore_outputs(exported(*flat_inputs), arguments)

    assert output["cache"] is runtime_cache
    torch.testing.assert_close(output["value"], value[..., :1, :])
    torch.testing.assert_close(runtime_cache.layers[0].keys, value)


def test_lazy_static_cache_initialization_survives_the_tensor_export_boundary():
    module = _InitializeCache().eval()
    key = torch.ones(1, 2, 1, 4)
    sample = ((key, _make_cache(), torch.tensor([0])), {})
    graph_spec = make_recorded_graph_spec(module, sample)
    exported = TorchExporter().export(module, sample, graph_spec).exported_program.module()

    runtime_cache = _make_cache()
    arguments = graph_spec.forward_signature.normalize((key, runtime_cache, torch.tensor([0])), {}).arguments
    flat_inputs = graph_spec.input_spec.flatten_tensors(arguments)
    output = graph_spec.restore_outputs(exported(*flat_inputs), arguments)

    torch.testing.assert_close(output[..., :1, :], key)
    torch.testing.assert_close(runtime_cache.layers[0].keys[..., :1, :], key)
    assert runtime_cache.layers[0].is_initialized
    assert runtime_cache.layers[0].device == key.device
    assert runtime_cache.layers[0].dtype == key.dtype
    assert runtime_cache.layers[0].max_batch_size == key.shape[0]
    assert runtime_cache.layers[0].num_heads == key.shape[1]
    assert runtime_cache.layers[0].head_dim == key.shape[3]


@pytest.mark.parametrize(
    "exporter",
    (
        pytest.param(TorchExporter(), id="inductor"),
        pytest.param(
            TorchExporter(use_auto=False, strict=False, fallback_to_dynamic_hints=True),
            id="torch-tensorrt",
        ),
    ),
)
def test_static_cache_export_accepts_a_dynamic_tensor_input(exporter):
    module = _ReadCache().eval()
    cache = _make_initialized_cache()
    sample = ((torch.ones(2, 4), cache), {})
    graph_spec = make_graph_spec(module.forward, sample, module(*sample[0]))
    update_input_spec(graph_spec, ((torch.ones(3, 4), cache), {}))

    exported = exporter.export(module, sample, graph_spec).exported_program.module()

    runtime_value = torch.ones(3, 4)
    arguments = graph_spec.forward_signature.normalize((runtime_value, _make_initialized_cache()), {}).arguments
    flat_inputs = graph_spec.input_spec.flatten_tensors(arguments)
    output = graph_spec.restore_outputs(exported(*flat_inputs), arguments)

    torch.testing.assert_close(output, runtime_value)
