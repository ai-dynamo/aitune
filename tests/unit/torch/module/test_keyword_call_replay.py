# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch
import wrapt

from aitune.global_context import BATCH_SIZE_KEY, global_context
from aitune.torch.libs.torch.torch_exporter import TorchExporter
from aitune.torch.module.call_compatibility import prefers_keyword_arguments
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import RecordingModule
from aitune.torch.module.tuned_module import TunedModule
from aitune.torch.task.correctness import check_inference_output_correctness


@wrapt.decorator
def _typed(wrapped, instance, args, kwargs):
    if args:
        raise TypeError("All arguments must be passed by kwargs only for typed methods")
    return wrapped(**kwargs)


class Typing:
    # Stand in for NeMo's optional base class without importing the ASR dependency stack.
    __module__ = "nemo.core.classes.common"


class TypedModel(torch.nn.Module, Typing):
    @cast(Any, _typed)
    def forward(self, input_signal, input_signal_length):
        return input_signal + input_signal_length[:, None]


@pytest.fixture
def recorded_model(tmp_path):
    model = TypedModel()
    recording = RecordingModule(model, "typed", cache_dir_resolver=lambda: tmp_path)
    inputs = {"input_signal": torch.ones(2, 4), "input_signal_length": torch.ones(2)}
    with global_context:
        global_context.set(BATCH_SIZE_KEY, 2)
        recording(**inputs)
        global_context.set(BATCH_SIZE_KEY, 3)
        recording(input_signal=torch.ones(3, 4), input_signal_length=torch.ones(3))
    graph = recording.graph_specs[0]
    assert graph.forward_signature.prefer_keyword_arguments
    return model, inputs, graph, recording.samples_for_graph_spec(graph)


def test_keyword_policy_covers_user_defined_subclasses():
    class CustomModel(TypedModel):
        pass

    assert prefers_keyword_arguments(CustomModel())


@pytest.mark.parametrize("module_name", [__name__, "nemo.collections.custom"])
def test_keyword_policy_does_not_match_unregistered_classes(module_name):
    unrelated = type("Typing", (torch.nn.Module,), {"__module__": module_name})
    assert not prefers_keyword_arguments(unrelated())


def test_recorded_keyword_call_survives_batching_and_correctness(recorded_model):
    model, inputs, graph, samples = recorded_model
    args, kwargs = samples[0]
    torch.testing.assert_close(model(*args, **kwargs), model(**inputs))
    args, kwargs = graph.make_batch(args, kwargs, batch_size=1)
    torch.testing.assert_close(model(*args, **kwargs), torch.full((1, 4), 2.0))
    check_inference_output_correctness(samples, graph, infer=model, name="typed")


@pytest.mark.parametrize("check_graph", [True, False])
def test_keyword_call_survives_tuned_dispatch_and_signature_restore(recorded_model, check_graph):
    model, inputs, graph, _ = recorded_model
    restored = GraphSpec.from_dict(graph.to_dict())
    backend = Mock()
    backend.infer.side_effect = model
    tuned = TunedModule(
        OrderedDict({restored.input_spec: backend}),
        "typed",
        restored.forward_signature,
        check_graph=check_graph,
    )
    torch.testing.assert_close(tuned(**inputs), model(**inputs))


@pytest.mark.parametrize("strict", [True, False])
def test_keyword_call_survives_torch_export(recorded_model, strict):
    model, inputs, graph, samples = recorded_model
    exported = TorchExporter(strict=strict).export(model, samples[0], graph, device="cpu")
    args, kwargs = exported.sample
    torch.testing.assert_close(exported.exported_program.module()(*args, **kwargs), model(**inputs))
