# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate keyword replay without depending on a third-party module family."""

import functools
from collections import OrderedDict
from unittest.mock import Mock

import pytest
import torch
import wrapt

from aitune.torch.libs.exporting import prepare_tensor_export
from aitune.torch.module.forward_signature import ForwardSignature
from aitune.torch.module.recording_module import RecordingModule
from aitune.torch.module.tuned_module import TunedModule
from aitune.torch.task.correctness import check_dynamic_shape_boundary_inference
from aitune.torch.utils.shapes import prepare_export_sample
from tests.utilities.helpers import make_graph_spec


def _require_keywords(forward):
    @functools.wraps(forward)
    def wrapped(self, *args, **kwargs):
        if args:
            raise TypeError("This forward decorator requires keyword arguments")
        return forward(self, **kwargs)

    return wrapped


class KeywordReplayModel(torch.nn.Module):
    @_require_keywords
    def forward(self, x):
        return x + 1


def test_recorded_sample_replays_keyword_decorated_module(tmp_path):
    model = KeywordReplayModel()
    recording = RecordingModule(model, "keyword_replay", cache_dir_resolver=lambda: tmp_path)
    x = torch.ones(2, 3)
    expected = recording(x=x)

    sample = recording.samples_for_graph_spec(recording.graph_specs[0])[0]

    torch.testing.assert_close(model(*sample[0], **sample[1]), expected)


@pytest.mark.parametrize("check_graph", [False, True])
def test_tuned_dispatch_replays_keyword_decorated_module(tmp_path, check_graph):
    model = KeywordReplayModel()
    recording = RecordingModule(model, "keyword_replay", cache_dir_resolver=lambda: tmp_path)
    x = torch.ones(2, 3)
    expected = recording(x=x)
    graph = recording.graph_specs[0]
    backend = Mock()
    backend.infer.side_effect = model
    tuned = TunedModule(
        OrderedDict({graph.input_spec: backend}),
        "keyword_replay",
        graph.forward_signature,
        check_graph=check_graph,
    )

    torch.testing.assert_close(tuned(x=x), expected)


@wrapt.decorator
def _typed_forward(wrapped, instance, args, kwargs):
    if args:
        raise TypeError("All arguments must be passed by kwargs only for typed methods")
    return wrapped(**kwargs)


class SpeechReplayModel(torch.nn.Module):
    @_typed_forward
    def forward(self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None):
        if processed_signal is not None:
            return processed_signal + 1, processed_signal_length
        return input_signal + 1, input_signal_length


@pytest.mark.parametrize("signal_kind", ["input", "processed"])
@pytest.mark.parametrize("check_graph", [False, True])
def test_speech_keyword_replay_survives_recording_export_and_dispatch(tmp_path, signal_kind, check_graph):
    model = SpeechReplayModel()
    recording = RecordingModule(model, "speech_replay", cache_dir_resolver=lambda: tmp_path)
    inputs = {f"{signal_kind}_signal": torch.ones(2, 3), f"{signal_kind}_signal_length": torch.ones(2)}
    expected = recording(**inputs)
    graph = recording.graph_specs[0]
    args, kwargs = recording.samples_for_graph_spec(graph)[0]

    torch.testing.assert_close(model(*args, **kwargs), expected)
    assert args == ()
    assert set(kwargs) == set(inputs)

    prepared_args, prepared_kwargs = prepare_export_sample((args, kwargs), graph)
    torch.testing.assert_close(model(*prepared_args, **prepared_kwargs), expected)
    preparation = prepare_tensor_export(model, (args, kwargs), graph)
    torch.testing.assert_close(preparation.adapter(*preparation.flat_inputs), expected)

    backend = Mock()
    backend.infer.side_effect = model
    restored_signature = ForwardSignature.from_dict(graph.forward_signature.to_dict())
    tuned = TunedModule(
        OrderedDict({graph.input_spec: backend}),
        "speech_replay",
        restored_signature,
        check_graph=check_graph,
    )
    torch.testing.assert_close(tuned(**inputs), expected)


def test_dynamic_boundary_checks_preserve_keyword_replay():
    model = KeywordReplayModel()
    sample = ((), {"x": torch.ones(2, 3)})
    graph = make_graph_spec(model.forward, sample, model(**sample[1]), batch_size=2)
    larger_sample = ((), {"x": torch.ones(4, 3)})
    larger_graph = make_graph_spec(model.forward, larger_sample, model(**larger_sample[1]), batch_size=4)
    # Both input and output metadata must describe the observed batch range.
    graph.update_shapes_seen(larger_graph.input_spec, larger_graph.output_spec)
    infer = Mock(side_effect=model)

    check_dynamic_shape_boundary_inference(sample, graph, infer, "keyword_replay")

    assert infer.call_count == 2
    assert all(call.args == () and set(call.kwargs) == {"x"} for call in infer.call_args_list)
    assert [call.kwargs["x"].shape for call in infer.call_args_list] == [(2, 3), (4, 3)]
