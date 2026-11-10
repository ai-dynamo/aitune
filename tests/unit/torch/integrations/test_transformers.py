# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Transformers cache integration."""

from copy import deepcopy

import pytest
import torch
import torch.nn as nn

from aitune.torch.libs.exporting import TorchExporter
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import RecordingModule
from aitune.torch.module.sample_metadata import SampleMetadata
from tests.utilities.helpers import make_graph_spec, make_recorded_graph_spec, update_input_spec

transformers = pytest.importorskip("transformers")
from transformers import EncoderDecoderCache, StaticCache, StaticLayer  # noqa: E402
from transformers.models.llama import LlamaConfig  # noqa: E402
from transformers.models.whisper import WhisperConfig  # noqa: E402


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
    recorded_keys = recorded_cache.layers[0].keys
    assert recorded_keys is not None
    value = torch.ones_like(recorded_keys)
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

    runtime_layer = runtime_cache.layers[0]
    assert isinstance(runtime_layer, StaticLayer)
    assert runtime_layer.keys is not None
    torch.testing.assert_close(output[..., :1, :], key)
    torch.testing.assert_close(runtime_layer.keys[..., :1, :], key)
    assert runtime_layer.is_initialized
    assert runtime_layer.device == key.device
    assert runtime_layer.dtype == key.dtype
    assert runtime_layer.max_batch_size == key.shape[0]
    assert runtime_layer.num_heads == key.shape[1]
    assert runtime_layer.head_dim == key.shape[3]


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


def _make_encoder_decoder_cache(*, initialized=True):
    config = WhisperConfig(
        d_model=8,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
        max_source_positions=4,
        max_target_positions=8,
    )
    caches = [StaticCache(config=config, max_cache_len=length) for length in (8, 4)]
    if initialized:
        for cache in caches:
            cache.early_initialization(
                batch_size=1, num_heads=2, head_dim=4, dtype=torch.float32, device=torch.device("cpu")
            )
    return EncoderDecoderCache(*caches)


class _WhisperCacheUpdate(nn.Module):
    """Follow Whisper's self-attention writes and one-time cross-attention initialization."""

    def forward(self, key, encoder_key, cache, position):
        keys, _ = cache.self_attention_cache.update(key, key * 2, 0, {"cache_position": position})
        if not cache.is_updated.get(0, False):
            cache.cross_attention_cache.update(
                encoder_key, encoder_key * 3, 0, {"cache_position": torch.arange(encoder_key.shape[-2])}
            )
            cache.is_updated[0] = True
        return {"keys": keys, "encoder_keys": cache.cross_attention_cache.layers[0].keys, "cache": cache}


def _assert_cache_matches(actual, expected):
    assert actual.is_updated == expected.is_updated
    for name in ("self_attention_cache", "cross_attention_cache"):
        actual_layer = getattr(actual, name).layers[0]
        expected_layer = getattr(expected, name).layers[0]
        assert actual_layer.is_initialized == expected_layer.is_initialized
        torch.testing.assert_close(actual_layer.keys, expected_layer.keys)
        torch.testing.assert_close(actual_layer.values, expected_layer.values)


def test_static_encoder_decoder_cache_metadata_includes_both_caches_and_readiness():
    cache = _make_encoder_decoder_cache()
    before = SampleMetadata.from_inputs({"cache": cache}, strict=True)

    assert {locator.path for locator, _ in before.tensor_data} == {
        ("cache", name, "layers", 0, field)
        for name in ("self_attention_cache", "cross_attention_cache")
        for field in ("keys", "values")
    }
    assert {locator.path: value for locator, value in before.other_data}[("cache", "is_updated", 0)] is False

    cache.is_updated[0] = True
    after = SampleMetadata.from_inputs({"cache": cache}, strict=True)

    # These phases take different Python branches even when all tensor shapes are identical.
    assert before != after


def test_static_encoder_decoder_cache_recording_separates_cross_attention_initialization(tmp_path):
    recording = RecordingModule(_WhisperCacheUpdate().eval(), "whisper-cache", cache_dir_resolver=lambda: tmp_path)
    cache = _make_encoder_decoder_cache()
    key = torch.ones(1, 2, 1, 4)
    encoder_key = torch.full((1, 2, 4, 4), 7.0)

    for position in (0, 1, 2):
        recording(key, encoder_key, cache, torch.tensor([position]))

    assert len(recording.graph_specs) == 2
    phase_flags = [
        {locator.path: value for locator, value in graph.input_spec.other_data}[("cache", "is_updated", 0)]
        for graph in recording.graph_specs
    ]
    assert phase_flags == [False, True]
    assert cache.is_updated[0] is True
    torch.testing.assert_close(cache.cross_attention_cache.layers[0].keys, encoder_key)


@pytest.mark.parametrize("initialized", [False, True], ids=["lazy", "preallocated"])
def test_static_encoder_decoder_cache_prefill_restores_nested_state_and_aliases(initialized):
    module = _WhisperCacheUpdate().eval()
    key = torch.ones(1, 2, 1, 4)
    encoder_key = torch.full((1, 2, 4, 4), 7.0)
    sample = ((key, encoder_key, _make_encoder_decoder_cache(initialized=initialized), torch.tensor([0])), {})
    graph_spec = make_recorded_graph_spec(module, sample, strict=True)
    exported = TorchExporter().export(module, sample, graph_spec).exported_program.module()
    runtime_cache = _make_encoder_decoder_cache(initialized=initialized)
    expected_cache = _make_encoder_decoder_cache(initialized=initialized)
    arguments = graph_spec.forward_signature.normalize(
        (key, encoder_key, runtime_cache, torch.tensor([0])), {}
    ).arguments

    expected = module(key, encoder_key, expected_cache, torch.tensor([0]))
    actual = graph_spec.restore_outputs(exported(*graph_spec.input_spec.flatten_tensors(arguments)), arguments)

    assert actual["cache"] is runtime_cache
    assert actual["keys"] is runtime_cache.self_attention_cache.layers[0].keys
    assert actual["encoder_keys"] is runtime_cache.cross_attention_cache.layers[0].keys
    torch.testing.assert_close(actual["keys"], expected["keys"])
    _assert_cache_matches(runtime_cache, expected_cache)
    # Export and recording must leave the supplied reference cache unchanged.
    assert sample[0][2].is_updated[0] is False


def test_static_encoder_decoder_cache_decode_round_trip_preserves_independent_callers(tmp_path):
    module = _WhisperCacheUpdate().eval()
    key = torch.ones(1, 2, 1, 4)
    encoder_key = torch.full((1, 2, 4, 4), 7.0)
    initial_cache = _make_encoder_decoder_cache()
    module(key, encoder_key, initial_cache, torch.tensor([0]))
    sample = ((key, encoder_key, initial_cache, torch.tensor([1])), {})
    graph_spec = make_recorded_graph_spec(module, sample, strict=True)
    program = TorchExporter().export(module, sample, graph_spec).exported_program
    path = tmp_path / "static-encoder-decoder.pt2"
    torch.export.save(program, path)
    exported = torch.export.load(path).module()
    graph_spec = GraphSpec.from_dict(graph_spec.to_dict())
    actual_caches = [deepcopy(initial_cache), deepcopy(initial_cache)]
    expected_caches = [deepcopy(initial_cache), deepcopy(initial_cache)]

    for cache_index, position, scale in ((0, 1, 2.0), (1, 1, 9.0), (0, 2, 3.0)):
        runtime_key = key * scale
        # Cross-attention must reuse its original state despite different encoder inputs during decode.
        changed_encoder_key = encoder_key * scale
        actual_cache = actual_caches[cache_index]
        args = (runtime_key, changed_encoder_key, actual_cache, torch.tensor([position]))
        arguments = graph_spec.forward_signature.normalize(args, {}).arguments
        expected = module(runtime_key, changed_encoder_key, expected_caches[cache_index], torch.tensor([position]))
        actual = graph_spec.restore_outputs(exported(*graph_spec.input_spec.flatten_tensors(arguments)), arguments)

        assert actual["cache"] is actual_cache
        torch.testing.assert_close(actual["keys"], expected["keys"])
        torch.testing.assert_close(actual["encoder_keys"], encoder_key)
        for actual_cache, expected_cache in zip(actual_caches, expected_caches, strict=True):
            _assert_cache_matches(actual_cache, expected_cache)
