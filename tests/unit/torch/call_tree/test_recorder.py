# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from aitune.torch.call_tree.methods import is_transparent_call, is_transparent_container, user_defined_methods
from aitune.torch.call_tree.recorder import OUTSIDE_ANY_CALL, CallRecorder
from tests.toy_models.call_tree_models import (
    PlainPipeline,
    SequentialSubclassToy,
    TwoRoleToy,
    VlaToy,
    vla_inputs,
)
from tests.unit.torch.call_tree.helpers import instrumented, record


def _labels(tree):
    return {n.label for n in tree.nodes.values()}


def test_same_module_from_two_callers_gives_two_nodes():
    model = TwoRoleToy()
    x = torch.randn(2, 8)

    tree = record(model, lambda: model(x))

    llm_nodes = [n for n in tree.nodes.values() if n.label == "root.llm.forward"]
    assert [n.key[-2] for n in llm_nodes] == ["root.prefill", "root.generate"]


def test_calls_are_counted_over_iterations_and_children_keep_first_call_order():
    model = VlaToy(steps=5)
    tokens, image = vla_inputs()

    tree = record(model, lambda: model(tokens, image), iterations=3)

    root = tree.nodes[("root.forward",)]
    assert tree.iterations == 3
    assert root.calls == 3
    dit = tree.nodes[("root.forward", "root.dit_forward")]
    assert dit.calls == 15
    children = [tree.nodes[k].label for k in root.children]
    assert children[:2] == ["root.vision.0.forward", "root.vision.1.forward"]
    assert children[-1] == "root.dit_forward"
    assert len(tree.sequences) == 3
    assert tree.sequences[0][0] == ("root.forward",)


def test_exact_containers_are_transparent_but_subclasses_are_recorded():
    model = VlaToy()
    tokens, image = vla_inputs()

    tree = record(model, lambda: model(tokens, image))

    assert "root.projector.forward" not in _labels(tree)
    assert ("root.forward", "root.projector.0.forward") in tree.nodes
    assert is_transparent_container(nn.Sequential())
    assert not is_transparent_container(SequentialSubclassToy())

    sub = SequentialSubclassToy()
    x = torch.randn(2, 8)
    tree = record(sub, lambda: sub(x))
    assert ("root.forward",) in tree.nodes


def test_helper_running_a_callable_argument_is_transparent():
    pipe = PlainPipeline()
    x = torch.randn(2, 8)

    tree = record(pipe, lambda: pipe.__call__(x))

    assert "root._apply_checkpoint" not in _labels(tree)
    assert ("root.encoder.forward",) in tree.nodes
    assert ("root.decode",) in tree.nodes
    assert is_transparent_call((len,))
    assert not is_transparent_call((nn.Linear(1, 1), torch.zeros(1), int))


def test_helper_receiving_its_callable_by_keyword_is_transparent():
    class KeywordCheckpoint(PlainPipeline):
        def __call__(self, x):
            return self.decode(self._apply_checkpoint(x=x, func=lambda y: self.encoder(y)))

    pipe = KeywordCheckpoint()
    x = torch.randn(2, 8)

    tree = record(pipe, lambda: pipe.__call__(x))

    assert "root._apply_checkpoint" not in _labels(tree)
    assert ("root.encoder.forward",) in tree.nodes
    assert is_transparent_call((), {"func": len})


def test_result_is_a_snapshot_that_later_recording_does_not_change():
    model = nn.Linear(2, 2)
    recorder = CallRecorder()

    def one_iteration():
        with recorder.recording():
            recorder.begin_iteration()
            frame = recorder.enter("root.forward", model, "forward", ())
            model(torch.randn(2, 2))
            recorder.exit(frame)
            recorder.end_iteration()

    one_iteration()
    first = recorder.result()
    one_iteration()

    assert first.nodes[("root.forward",)].calls == 1
    assert first.iterations == 1
    assert recorder.result().nodes[("root.forward",)].calls == 2


def test_second_recorder_cannot_record_while_one_is_active():
    first, second = CallRecorder(), CallRecorder()
    original_apply = nn.Module._apply

    with first.recording():
        with pytest.raises(RuntimeError, match="already recording"):
            with second.recording():
                pass
        assert nn.Module._apply is not original_apply
    assert nn.Module._apply is original_apply
    with second.recording():
        pass


def test_apply_probe_records_callers_reaching_modules_from_outside():
    model = SequentialSubclassToy()
    x = torch.randn(2, 8)

    tree = record(model, lambda: model.encode(x))

    callers = tree.external_apply[id(model.get_submodule("0.layers.0.proj"))]
    assert callers == {"root.encode"}


def test_apply_outside_any_recorded_call():
    recorder = CallRecorder()
    module = nn.Linear(1, 1)
    with recorder.recording():
        module.to("cpu")
    assert recorder.result().external_apply[id(module)] == {"<unrecorded: test_apply_outside_any_recorded_call>"}


def test_apply_from_a_library_style_to_override_names_the_real_caller():
    class Wrapper(nn.Linear):
        def to(self, *args, **kwargs):
            return super().to(*args, **kwargs)

    def caller(module):
        module.to("cpu")

    recorder = CallRecorder()
    module = Wrapper(1, 1)
    with recorder.recording():
        caller(module)

    expected = "<unrecorded: test_apply_from_a_library_style_to_override_names_the_real_caller.<locals>.caller>"
    assert recorder.result().external_apply[id(module)] == {expected}


def test_apply_with_only_torch_frames_falls_back_to_outside_any_call(mocker):
    mocker.patch("aitune.torch.call_tree.recorder.sys._getframe", return_value=None)
    recorder = CallRecorder()
    module = nn.Linear(1, 1)
    with recorder.recording():
        module.to("cpu")
    assert recorder.result().external_apply[id(module)] == {OUTSIDE_ANY_CALL}


def test_apply_probe_records_once_for_one_recursive_to(mocker):
    spy = mocker.spy(nn.Module, "modules")
    model = SequentialSubclassToy()  # 3 levels: root -> layers -> block -> proj
    x = torch.randn(2, 8)

    tree = record(model, lambda: model.encode(x), iterations=1)

    assert spy.call_count == 1
    callers = tree.external_apply[id(model.get_submodule("0.layers.0.proj"))]
    assert callers == {"root.encode"}
    reached = set(tree.external_apply)
    assert {id(module) for module in model.modules()} <= reached


def test_apply_probe_respects_recurse_false():
    recorder = CallRecorder()
    model = SequentialSubclassToy()
    with recorder.recording():
        model._apply(lambda tensor: tensor, False)
    assert set(recorder.result().external_apply) == {id(model)}


def test_apply_probe_is_restored_after_an_exception():
    original = nn.Module._apply
    recorder = CallRecorder()
    with pytest.raises(RuntimeError), recorder.recording():
        assert nn.Module._apply is not original
        raise RuntimeError("boom")
    assert nn.Module._apply is original
    assert not recorder.active


def test_nothing_is_recorded_outside_recording():
    model = TwoRoleToy()
    recorder = CallRecorder()
    with instrumented(recorder, model):
        model(torch.randn(2, 8))
    assert recorder.result().nodes == {}


def test_cpu_timing_is_recorded(mocker):
    mocker.patch("aitune.torch.call_tree.recorder.torch.cuda.is_available", return_value=False)
    model = TwoRoleToy()
    x = torch.randn(2, 8)

    tree = record(model, lambda: model(x))

    assert tree.nodes[("root.forward",)].time_ms > 0
    assert tree.total_ms == tree.nodes[("root.forward",)].time_ms


def _record_one_call(recorder, model, x):
    with instrumented(recorder, model), recorder.recording():
        recorder.begin_iteration()
        model(x)
        recorder.end_iteration()


def test_result_returns_the_same_times_twice_and_keeps_accumulating(mocker):
    mocker.patch("aitune.torch.call_tree.recorder.torch.cuda.is_available", return_value=False)
    model = TwoRoleToy()
    recorder = CallRecorder()
    x = torch.randn(2, 8)

    _record_one_call(recorder, model, x)
    first = recorder.result().nodes[("root.forward",)].time_ms
    assert first > 0
    assert recorder.result().nodes[("root.forward",)].time_ms == first

    _record_one_call(recorder, model, x)
    assert recorder.result().nodes[("root.forward",)].time_ms > first


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA events require a GPU")
def test_result_returns_the_same_times_twice_with_cuda_events():
    model = TwoRoleToy().cuda()
    recorder = CallRecorder()

    _record_one_call(recorder, model, torch.randn(2, 8, device="cuda"))
    first = recorder.result().nodes[("root.forward",)].time_ms

    assert first > 0
    assert recorder.result().nodes[("root.forward",)].time_ms == first


def test_user_defined_methods_skip_library_and_base_methods():
    assert user_defined_methods(VlaToy()) == ["dit_forward"]
    assert user_defined_methods(nn.Linear(1, 1)) == []
    assert user_defined_methods(PlainPipeline()) == ["decode", "_apply_checkpoint"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA events require a GPU")
def test_flush_resolves_pending_events_into_totals():
    model = nn.Linear(2, 2).cuda()
    recorder = CallRecorder()

    with recorder.recording():
        recorder.begin_iteration()
        frame = recorder.enter("root.forward", model, "forward", ())
        model(torch.randn(2, 2, device="cuda"))
        recorder.exit(frame)
        recorder.end_iteration()

    assert recorder._events
    recorder.flush()
    assert not recorder._events

    assert recorder.result().nodes[("root.forward",)].time_ms > 0
    assert recorder.result().nodes[("root.forward",)].time_ms == recorder.result().nodes[("root.forward",)].time_ms
