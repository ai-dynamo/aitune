# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test the Transformers integration for convolution padding caches passed between generation steps."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from aitune.torch.integrations import device as device_integration
from aitune.torch.module.locator import Locator
from aitune.torch.module.recording_module import RecordingModule
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.task.correctness import check_inference_output_correctness

_TRANSFORMERS_INTEGRATION_PATH = Path(__file__).parents[4] / "aitune" / "torch" / "integrations" / "transformers.py"
_VOXTRAL_REALTIME_MODULE = "transformers.models.voxtral_realtime.modeling_voxtral_realtime"
_CHANNELS = 2
_LEFT_PAD = 2


class _PaddingCacheLayer:
    """Follow Voxtral Realtime's cache layer: lazily allocated padding state, updated in place."""

    def __init__(self):
        self.cache: torch.Tensor | None = None
        self.is_initialized = False

    def update(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.is_initialized:
            self.cache = torch.zeros(*hidden_states.shape[:2], _LEFT_PAD, dtype=hidden_states.dtype)
            self.is_initialized = True
        assert self.cache is not None
        current_cache = self.cache.clone()
        self.cache.copy_(hidden_states[..., -_LEFT_PAD:])
        return current_cache


class _PaddingCache:
    """Follow Voxtral Realtime's padding cache: one layer per convolution and no value equality."""

    def __init__(self):
        self.layers = {}

    def update(self, hidden_states: torch.Tensor, cache_key: str) -> torch.Tensor:
        layer = self.layers.setdefault(cache_key, _PaddingCacheLayer())
        return torch.cat([layer.update(hidden_states), hidden_states], dim=-1)


class _StreamingConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(_CHANNELS, _CHANNELS, kernel_size=_LEFT_PAD + 1)

    def forward(self, x: torch.Tensor, padding_cache: _PaddingCache) -> torch.Tensor:
        return self.conv(padding_cache.update(x, "conv"))


@pytest.fixture
def padding_cache_integration(monkeypatch):
    """Run the Transformers integration against stand-ins for the Voxtral Realtime cache classes."""
    voxtral_realtime = types.ModuleType(_VOXTRAL_REALTIME_MODULE)
    voxtral_realtime.VoxtralRealtimeConv1dPaddingCache = _PaddingCache
    voxtral_realtime.VoxtralRealtimeConv1dCacheLayer = _PaddingCacheLayer
    monkeypatch.setitem(sys.modules, _VOXTRAL_REALTIME_MODULE, voxtral_realtime)
    # The device property is already patched when aitune.torch is imported; do not wrap it again.
    monkeypatch.setattr(device_integration, "patch_device_property", lambda cls: None)

    spec = importlib.util.spec_from_file_location(
        "_aitune_torch_integrations_transformers_test", _TRANSFORMERS_INTEGRATION_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    try:
        yield
    finally:
        Locator.unregister_user_type(_PaddingCache)
        Locator.unregister_user_type(_PaddingCacheLayer)


def _primed_cache(module: nn.Module, batch_size: int) -> _PaddingCache:
    """Return a padding cache initialized by a first streaming chunk, as after prefill."""
    cache = _PaddingCache()
    with torch.no_grad():
        module(torch.randn(batch_size, _CHANNELS, 3), cache)
    return cache


def test_padding_cache_metadata_tracks_tensors_instead_of_object_identity(padding_cache_integration):
    module = _StreamingConv().eval()
    first, second = _primed_cache(module, 1), _primed_cache(module, 1)

    first_metadata = SampleMetadata.from_inputs({"padding_cache": first}, strict=True)
    second_metadata = SampleMetadata.from_inputs({"padding_cache": second}, strict=True)

    assert first_metadata.other_data == ()
    assert [locator.path for locator, _ in first_metadata.tensor_data] == [("padding_cache", "layers", "conv", "cache")]
    assert first_metadata == second_metadata
    assert hash(first_metadata) == hash(second_metadata)


def test_padding_cache_calls_share_a_graph_and_pass_input_mutation_check(padding_cache_integration, tmp_path):
    module = _StreamingConv().eval()
    recording = RecordingModule(module, "streaming-conv", cache_dir_resolver=lambda: tmp_path)
    with torch.no_grad():
        for batch_size in (1, 2):
            # Each request owns a new cache, as separate generate() calls do.
            cache = _primed_cache(module, batch_size)
            for _ in range(2):
                recording(torch.randn(batch_size, _CHANNELS, 3), cache)

    assert len(recording.graph_specs) == 1
    graph_spec = recording.graph_specs[0]
    assert graph_spec.post_input_spec is not None

    # The check runs on deep copies of recorded samples, so the cache passed to the backend is a new object.
    check_inference_output_correctness(
        recording.samples_for_graph_spec(graph_spec).iter_samples(),
        graph_spec,
        infer=module,
        name="streaming-conv.0.eager",
    )


def test_voxtral_realtime_padding_cache_metadata_tracks_tensors():
    voxtral_realtime = pytest.importorskip(_VOXTRAL_REALTIME_MODULE)
    conv = voxtral_realtime.VoxtralRealtimeCausalConv1d(_CHANNELS, _CHANNELS, kernel_size=3, cache_key="conv1").eval()
    caches = [voxtral_realtime.VoxtralRealtimeConv1dPaddingCache() for _ in range(2)]
    with torch.no_grad():
        for cache in caches:
            conv(torch.randn(1, _CHANNELS, 4), padding_cache=cache)

    metadata = [SampleMetadata.from_inputs({"padding_cache": cache}, strict=True) for cache in caches]

    assert metadata[0].other_data == ()
    assert [locator.path for locator, _ in metadata[0].tensor_data] == [("padding_cache", "layers", "conv1", "cache")]
    assert metadata[0] == metadata[1]
