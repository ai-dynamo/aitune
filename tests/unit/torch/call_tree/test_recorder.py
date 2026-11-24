# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from collections import UserDict
from collections.abc import Mapping

import pytest
import torch
from torch import nn

from aitune.torch.call_tree.callers import OUTSIDE_ANY_CALL
from aitune.torch.call_tree.methods import is_transparent_call, is_transparent_container, user_defined_methods
from aitune.torch.call_tree.nodes import CallSignature, Leaf, TensorLeaf
from aitune.torch.call_tree.recorder import CallRecorder
from aitune.torch.call_tree.weights.accesses import WeightAccess, WeightAccessLog, storage_key
from tests.toy_models.call_tree_models import (
    PlainPipeline,
    SequentialSubclassToy,
    TwoRoleToy,
    VlaToy,
    vla_inputs,
)
from tests.unit.torch.call_tree.helpers import instrumented, invisible_to_recorder, record


def _labels(tree):
    return {n.label for n in tree.nodes.values()}


def test_same_module_from_two_callers_gives_two_nodes():
    model = TwoRoleToy()
    x = torch.randn(2, 8)

    tree = record(model, lambda: model(x))

    llm_nodes = [n for n in tree.nodes.values() if n.label == "root.llm.forward"]
    assert [n.path[-2] for n in llm_nodes] == ["root.prefill", "root.generate"]
    assert tree.nodes_of(model.llm) == llm_nodes


def test_nodes_of_finds_a_modules_nodes_by_identity_and_method():
    model = VlaToy(steps=2)
    tokens, image = vla_inputs()

    tree = record(model, lambda: model(tokens, image))

    assert [n.path_str for n in tree.nodes_of(model.dit)] == ["forward > dit_forward > dit.forward"]
    assert [n.path_str for n in tree.nodes_of(model, "dit_forward")] == ["forward > dit_forward"]
    assert tree.nodes_of(nn.Linear(2, 2)) == []


def test_calls_are_counted_over_iterations_and_children_keep_first_call_order():
    model = VlaToy(steps=5)
    tokens, image = vla_inputs()

    tree = record(model, lambda: model(tokens, image), iterations=3)

    root = tree.nodes[("root.forward",)]
    assert tree.iterations == 3
    assert root.calls == 3
    dit = tree.nodes[("root.forward", "root.dit_forward")]
    assert dit.calls == 15
    children = [tree.nodes[p].label for p in root.children]
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
        with recorder:
            recorder.begin_iteration()
            frame = recorder.begin_call("root.forward", model, "forward", ())
            model(torch.randn(2, 2))
            recorder.end_call(frame)
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

    with first:
        with pytest.raises(RuntimeError, match="already recording"):
            with second:
                pass
        assert nn.Module._apply is not original_apply
    assert nn.Module._apply is original_apply
    with second:
        pass


def test_apply_probe_records_move_accesses_reaching_modules_from_outside():
    model = SequentialSubclassToy()
    x = torch.randn(2, 8)

    tree = record(model, lambda: model.encode(x))

    assert WeightAccess("encode", "move") in tree.weight_accesses["0.layers.0.proj.weight"]


def test_apply_outside_any_recorded_call():
    recorder = CallRecorder()
    module = nn.Linear(1, 1)
    recorder.register_module("root", module)
    with recorder:
        module.to("cpu")
    assert recorder.result().weight_accesses["weight"] == {
        WeightAccess("<unrecorded: test_apply_outside_any_recorded_call>", "move")
    }


def test_apply_from_a_library_style_to_override_names_the_real_caller():
    class Wrapper(nn.Linear):
        def to(self, *args, **kwargs):
            return super().to(*args, **kwargs)

    def caller(module):
        module.to("cpu")

    recorder = CallRecorder()
    module = Wrapper(1, 1)
    recorder.register_module("root", module)
    with recorder:
        caller(module)

    expected = "<unrecorded: test_apply_from_a_library_style_to_override_names_the_real_caller.<locals>.caller>"
    assert recorder.result().weight_accesses["weight"] == {WeightAccess(expected, "move")}


def test_apply_with_only_torch_frames_falls_back_to_outside_any_call(mocker):
    mocker.patch("aitune.torch.call_tree.callers.sys._getframe", return_value=None)
    recorder = CallRecorder()
    module = nn.Linear(1, 1)
    recorder.register_module("root", module)
    with recorder:
        module.to("cpu")
    assert recorder.result().weight_accesses["weight"] == {WeightAccess(OUTSIDE_ANY_CALL, "move")}


def test_apply_probe_records_once_for_one_recursive_to(mocker):
    spy = mocker.spy(nn.Module, "modules")
    model = SequentialSubclassToy()  # 3 levels: root -> layers -> block -> proj
    x = torch.randn(2, 8)

    tree = record(model, lambda: model.encode(x), iterations=1)

    assert spy.call_count == 1
    assert WeightAccess("encode", "move") in tree.weight_accesses["0.layers.0.proj.weight"]
    assert all(WeightAccess("encode", "move") in accesses for accesses in tree.weight_accesses.values())


def test_apply_probe_respects_recurse_false():
    class RootWithWeight(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(4))
            self.child = nn.Linear(4, 4)

    recorder = CallRecorder()
    model = RootWithWeight()
    recorder.register_module("root", model)
    with recorder:
        model._apply(lambda tensor: tensor, False)

    accesses = recorder.result().weight_accesses
    assert accesses["w"] == {WeightAccess("<unrecorded: test_apply_probe_respects_recurse_false>", "move")}
    assert "child.weight" not in accesses
    assert "child.bias" not in accesses


def test_a_move_records_only_the_move_for_its_caller():
    """The probe's own storage lookups are not recorded as reads."""
    recorder = CallRecorder()
    module = nn.Linear(4, 4)
    recorder.register_module("root", module)

    def move():
        module.to("cpu")

    with recorder:
        move()

    expected = WeightAccess("<unrecorded: test_a_move_records_only_the_move_for_its_caller.<locals>.move>", "move")
    accesses = recorder.result().weight_accesses
    assert accesses["weight"] == {expected}
    assert accesses["bias"] == {expected}


def test_apply_probe_is_restored_after_an_exception():
    original = nn.Module._apply
    recorder = CallRecorder()
    with pytest.raises(RuntimeError):
        with recorder:
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
    with instrumented(recorder, model), recorder:
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

    with recorder:
        recorder.begin_iteration()
        frame = recorder.begin_call("root.forward", model, "forward", ())
        model(torch.randn(2, 2, device="cuda"))
        recorder.end_call(frame)
        recorder.end_iteration()

    assert recorder._events
    recorder.flush()
    assert not recorder._events

    assert recorder.result().nodes[("root.forward",)].time_ms > 0
    assert recorder.result().nodes[("root.forward",)].time_ms == recorder.result().nodes[("root.forward",)].time_ms


def test_end_event_is_recorded_on_the_start_device_when_the_call_changes_device(mocker):
    events = []

    class FakeEvent:
        def __init__(self, *args, **kwargs):
            events.append(self)
            self.stream = None

        def record(self, stream=None):
            self.stream = stream

        def synchronize(self):
            pass

        def elapsed_time(self, end):
            return 2.0

    mocker.patch("aitune.torch.call_tree.recorder.torch.cuda.Event", FakeEvent)
    current_device = mocker.patch("aitune.torch.call_tree.recorder.torch.cuda.current_device", return_value=0)
    mocker.patch(
        "aitune.torch.call_tree.recorder.torch.cuda.current_stream",
        side_effect=lambda device=None: f"stream{device}",
    )
    recorder = CallRecorder()
    recorder._use_cuda = True
    model = nn.Linear(2, 2)

    with recorder:
        recorder.begin_iteration()
        frame = recorder.begin_call("root.forward", model, "forward", ())
        current_device.return_value = 1  # the call changed the current device
        recorder.end_call(frame)
        recorder.end_iteration()

    tree = recorder.result()
    start, end = events
    assert start.stream is None
    assert end.stream == "stream0"
    assert tree.nodes[("root.forward",)].time_ms == 2.0


def test_flush_synchronizes_every_end_event(mocker):
    events = []

    class FakeEvent:
        def __init__(self, *args, **kwargs):
            events.append(self)
            self.synchronized = False

        def record(self, stream=None):
            pass

        def synchronize(self):
            self.synchronized = True

        def elapsed_time(self, end):
            if not end.synchronized:
                raise RuntimeError("Both events must be completed")
            return 1.0

    mocker.patch("aitune.torch.call_tree.recorder.torch.cuda.Event", FakeEvent)
    mocker.patch("aitune.torch.call_tree.recorder.torch.cuda.current_device", return_value=0)
    mocker.patch(
        "aitune.torch.call_tree.recorder.torch.cuda.current_stream",
        side_effect=lambda device=None: f"stream{device}",
    )
    cuda_synchronize = mocker.patch("aitune.torch.call_tree.recorder.torch.cuda.synchronize")
    recorder = CallRecorder()
    recorder._use_cuda = True
    model = nn.Linear(2, 2)

    with recorder:
        recorder.begin_iteration()
        for _ in range(2):
            recorder.end_call(recorder.begin_call("root.forward", model, "forward", ()))
        recorder.end_iteration()
        recorder.flush()

    assert recorder._nodes[("root.forward",)].time_ms == 2.0
    cuda_synchronize.assert_not_called()


def test_accesses_are_recorded_only_inside_recording():
    """Ops on weights outside the recorder's ``with`` block leave no weight_accesses."""
    model = torch.nn.Linear(4, 4)
    recorder = CallRecorder()
    with instrumented(recorder, model):
        model.bias.sum()  # outside recording: not recorded
        with recorder:
            recorder.begin_iteration()
            model.weight.sum()
            recorder.end_iteration()
        model.bias.sum()  # outside again
    tree = recorder.result()
    assert "weight" in tree.weight_accesses
    assert "bias" not in tree.weight_accesses


def test_access_mode_is_removed_after_recording_raises():
    """An exception inside the recorder's ``with`` block still restores torch_function state."""
    model = torch.nn.Linear(4, 4)
    recorder = CallRecorder()
    recorder.register_module("root", model)

    def record_then_raise():
        with recorder:
            recorder.begin_iteration()
            model.weight.sum()
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        record_then_raise()
    # if the mode leaked, this op would be recorded
    before = recorder.result().weight_accesses
    model.weight.sum()
    assert recorder.result().weight_accesses == before


def test_ops_inside_apply_are_not_recorded():
    """.to() internals produce no reads; the probe records the move itself as a ``"move"`` access."""
    model = torch.nn.Linear(4, 4)
    recorder = CallRecorder()
    with instrumented(recorder, model), recorder:
        recorder.begin_iteration()
        model.to("cpu")
        recorder.end_iteration()
    expected = WeightAccess("<unrecorded: test_ops_inside_apply_are_not_recorded>", "move")
    assert recorder.result().weight_accesses == {"weight": {expected}, "bias": {expected}}


def test_a_failing_access_mode_leaves_no_apply_probe_behind(mocker):
    original_apply = nn.Module._apply
    mocker.patch("aitune.torch.call_tree.weights.accesses.WeightAccessMode.__init__", side_effect=RuntimeError("boom"))
    recorder = CallRecorder()

    with pytest.raises(RuntimeError, match="boom"):
        with recorder:
            pass

    assert nn.Module._apply is original_apply
    assert CallRecorder._recording_now is None


def test_weight_accesses_accumulate_over_recording_passes():
    linear = nn.Linear(8, 8)
    recorder = CallRecorder()
    recorder.register_module("root", linear)

    with recorder:
        torch.nn.functional.linear(torch.randn(2, 8), linear.weight)
    with recorder:
        pass

    assert {kind for _, kind in recorder.result().weight_accesses["weight"]} == {"read"}


def test_reads_after_a_cast_during_recording_are_tracked_under_the_new_storage():
    linear = nn.Linear(8, 8).double()
    recorder = CallRecorder()
    recorder.register_module("root", linear)

    with recorder:
        linear.float()  # new storage for the weight
        torch.nn.functional.linear(torch.randn(2, 8), linear.weight)

    assert {kind for _, kind in recorder.result().weight_accesses["weight"]} == {"read", "move"}


def test_access_of_a_weight_cast_through_an_unregistered_child_is_tracked():
    linear = nn.Linear(4, 4)
    seq = nn.Sequential(linear)
    recorder = CallRecorder()
    recorder.register_module("root", seq)

    with recorder:
        seq[0].double()  # unregistered submodule is cast
        torch.nn.functional.linear(torch.randn(2, 4).double(), linear.weight)

    assert set(recorder.result().weight_accesses) == {"0.weight", "0.bias"}
    assert {kind for _, kind in recorder.result().weight_accesses["0.weight"]} == {"read", "move"}


def test_a_moved_weight_stops_mapping_under_its_old_storage():
    linear = nn.Linear(4, 4)
    log = WeightAccessLog()
    log.start([("root", linear)])
    old_storage = linear.weight.untyped_storage()  # kept alive so its address is not reused
    old_key = storage_key(linear.weight)

    linear.double()
    log.retrack()

    assert linear.weight.untyped_storage().data_ptr() != old_storage.data_ptr()
    access = WeightAccess("probe", "read")
    log.record(old_key, access)
    assert log.snapshot() == {}
    log.record(storage_key(linear.weight), access)
    assert log.snapshot() == {"weight": {access}}


def test_a_tied_weight_retracks_with_the_moved_sibling():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(8, 4)
            self.head = nn.Linear(4, 8)
            self.head.weight = self.embed.weight

    model = Model()
    log = WeightAccessLog()
    log.start([("root", model)])
    old_storage = model.embed.weight.untyped_storage()  # kept alive so its address is not reused

    old_key = storage_key(model.embed.weight)
    model.head.double()  # gives the tied Parameter new storage, outside embed's subtree
    log.retrack()

    assert model.embed.weight.untyped_storage().data_ptr() != old_storage.data_ptr()
    access = WeightAccess("probe", "read")
    log.record(old_key, access)
    assert log.snapshot() == {}
    log.record(storage_key(model.head.weight), access)
    assert log.snapshot() == {"embed.weight": {access}, "head.weight": {access}}


def test_a_tied_weight_is_read_through_both_names_after_a_move():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(8, 4)
            self.head = nn.Linear(4, 8)
            self.head.weight = self.embed.weight

    model = Model()
    recorder = CallRecorder()
    recorder.register_module("root", model)
    old_storage = model.embed.weight.untyped_storage()  # kept alive so its address is not reused

    with recorder:
        model.head.double()  # partial move: the tied Parameter gets new storage outside embed's subtree
        torch.nn.functional.embedding(torch.tensor([1]), model.embed.weight)

    assert model.embed.weight.untyped_storage().data_ptr() != old_storage.data_ptr()

    accesses = recorder.result().weight_accesses
    assert {kind for _, kind in accesses["embed.weight"]} == {"read", "move"}
    assert {kind for _, kind in accesses["head.weight"]} == {"read", "move"}


def test_an_untracked_parameter_miss_retracks_only_once(mocker):
    """A free Parameter not under any registered module triggers one miss-triggered re-scan."""
    linear = nn.Linear(4, 4)
    recorder = CallRecorder()
    recorder.register_module("root", linear)
    free = nn.Parameter(torch.randn(4))
    calls = 0

    real_rebuild = WeightAccessLog._rebuild_storage_map

    def counting(self):
        nonlocal calls
        calls += 1
        return real_rebuild(self)

    mocker.patch.object(WeightAccessLog, "_rebuild_storage_map", counting)
    with recorder:
        calls = 0  # ignore the start() re-scan
        free.sum()
        (free + 1).sum()

    assert calls == 1


def test_two_untracked_parameters_used_alternately_retrack_once_each(mocker):
    """Miss-triggered re-scans keep the untracked cache: each free Parameter re-scans only once."""
    linear = nn.Linear(4, 4)
    recorder = CallRecorder()
    recorder.register_module("root", linear)
    a = nn.Parameter(torch.randn(4))
    b = nn.Parameter(torch.randn(4))
    calls = 0

    real_rebuild = WeightAccessLog._rebuild_storage_map

    def counting(self):
        nonlocal calls
        calls += 1
        return real_rebuild(self)

    mocker.patch.object(WeightAccessLog, "_rebuild_storage_map", counting)
    with recorder:
        calls = 0
        for _ in range(5):
            a.sum()
            b.sum()

    assert calls == 2


def test_a_tied_weight_is_recorded_under_each_of_its_names():
    linear = nn.Linear(8, 8)
    linear.register_buffer("weight_alias", linear.weight)
    recorder = CallRecorder()
    recorder.register_module("root", linear)

    with recorder:
        torch.nn.functional.linear(torch.randn(2, 8), linear.weight)

    accesses = recorder.result().weight_accesses
    assert accesses["weight_alias"] == accesses["weight"]
    assert {kind for _, kind in accesses["weight"]} == {"read"}


def test_a_parameter_swapped_outside_apply_does_not_keep_its_old_storage():
    """After ``param.data = ...``, an op on a kept reference to the old storage is not a read of the weight."""
    unit = nn.Linear(8, 8)
    model = nn.Sequential(unit)
    x = torch.randn(2, 8)

    def infer():
        unit(x)
        with invisible_to_recorder():
            old = unit.weight.data
            unit.weight.data = old.clone()

        def touch_old():
            old.sum()

        touch_old()
        torch.nn.functional.linear(x, unit.weight)

    tree = record(model, infer)

    accesses = tree.weight_accesses["0.weight"]
    touch = f"<unrecorded: {infer.__qualname__}.<locals>.touch_old>"
    assert WeightAccess(touch, "read") not in accesses
    assert WeightAccess(f"<unrecorded: {infer.__qualname__}>", "read") in accesses


def test_a_buffer_swapped_outside_apply_does_not_keep_its_old_storage():
    """After a buffer is replaced in ``_buffers``, an op on the old storage is not a read of the buffer."""
    unit = nn.Linear(8, 8)
    unit.register_buffer("scale", torch.ones(8))
    model = nn.Sequential(unit)
    x = torch.randn(2, 8)

    def infer():
        unit(x)
        with invisible_to_recorder():
            old = unit.get_buffer("scale")
            unit._buffers["scale"] = old.clone()

        def touch_old():
            old.sum()

        touch_old()

    tree = record(model, infer)

    touch = f"<unrecorded: {infer.__qualname__}.<locals>.touch_old>"
    assert WeightAccess(touch, "read") not in tree.weight_accesses.get("0.scale", set())


def test_a_tied_weight_replaced_under_one_name_only_credits_the_other_name():
    """A tied name whose attribute got new storage is not credited for ops on the sibling name's old storage."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 4)
            self.b.weight = self.a.weight

        def forward(self, x):
            return x

    model = Model()

    def infer():
        with invisible_to_recorder():
            model.b._parameters["weight"] = nn.Parameter(model.b.weight.detach().clone())
        model.a.weight.sum()

    tree = record(model, infer)

    read = WeightAccess(f"<unrecorded: {infer.__qualname__}>", "read")
    assert read in tree.weight_accesses["a.weight"]
    assert read not in tree.weight_accesses.get("b.weight", set())


def test_a_tied_weight_removed_under_one_name_only_credits_the_other_name():
    """A tied name whose attribute was set to ``None`` is not credited for ops on the sibling name's storage."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 4)
            self.b.weight = self.a.weight

        def forward(self, x):
            return x

    model = Model()

    def infer():
        with invisible_to_recorder():
            model.b._parameters["weight"] = None
        model.a.weight.sum()

    tree = record(model, infer)

    read = WeightAccess(f"<unrecorded: {infer.__qualname__}>", "read")
    assert read in tree.weight_accesses["a.weight"]
    assert read not in tree.weight_accesses.get("b.weight", set())


def test_bytes_in_counts_tensors_in_nested_args_and_kwargs():
    class Model(nn.Module):
        def forward(self, x, extra=None):
            return x

    model = Model()
    a, b, c = torch.randn(4, 8), torch.randn(2, 8), torch.randn(8)
    expected = a.numel() * a.element_size() + b.numel() * b.element_size() + c.numel() * c.element_size()

    tree = record(model, lambda: model([a, b], extra={"t": c}), iterations=1)

    node = tree.nodes[("root.forward",)]
    assert node.bytes_in == expected
    assert node.bytes_in / node.calls == expected


def test_bytes_in_counts_a_tensor_passed_twice_once():
    class Model(nn.Module):
        def forward(self, x, extra=None):
            return x

    model = Model()
    t = torch.randn(4, 8)

    tree = record(model, lambda: model(t, extra={"again": t}), iterations=1)

    assert tree.nodes[("root.forward",)].bytes_in == t.numel() * t.element_size()


def test_bytes_ignores_non_tensor_objects():
    class Model(nn.Module):
        def forward(self, x, cache=None):
            return x, cache  # returns the untracked object too

    model = Model()
    t = torch.randn(4, 8)
    obj = object()

    tree = record(model, lambda: model(t, cache=obj), iterations=1)

    node = tree.nodes[("root.forward",)]
    assert node.bytes_in == t.numel() * t.element_size()
    assert node.bytes_out == t.numel() * t.element_size()  # the (x, cache) tuple counts only x


def test_bytes_out_counts_the_tensors_of_a_model_output():
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast

    class Model(nn.Module):
        def forward(self, x):
            output = BaseModelOutputWithPast(last_hidden_state=x)
            # Set the cache through ModelOutput's mapping API; newer constructor annotations omit this field.
            output["past_key_values"] = DynamicCache()
            return output

    model = Model()
    t = torch.randn(4, 8)

    tree = record(model, lambda: model(t), iterations=1)

    node = tree.nodes[("root.forward",)]
    assert node.bytes_out == t.numel() * t.element_size()


def test_bytes_sum_over_calls_and_iterations():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(8, 8)

        def forward(self, x):
            return self.head(self.head(x))

    model = Model()
    x = torch.randn(4, 8)
    per_call = x.numel() * x.element_size()
    iterations = 3

    tree = record(model, lambda: model(x), iterations=iterations)

    [node] = tree.nodes_of(model.head)
    calls = 2 * iterations  # forward calls head twice per iteration
    assert node.calls == calls
    assert node.bytes_in == calls * per_call
    assert node.bytes_out == calls * per_call  # Linear returns a same-shaped tensor here


def test_bytes_in_counts_a_call_that_raises_but_not_bytes_out():
    class Model(nn.Module):
        def forward(self, x):
            raise RuntimeError("boom")

    model = Model()
    x = torch.randn(4, 8)

    recorder = CallRecorder()
    with instrumented(recorder, model), recorder:
        recorder.begin_iteration()
        with pytest.raises(RuntimeError, match="boom"):
            model(x)
        recorder.end_iteration()
    tree = recorder.result()

    node = tree.nodes[("root.forward",)]
    assert node.bytes_in == x.numel() * x.element_size()
    assert node.bytes_out == 0


def test_the_bytes_walk_records_no_weight_access():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, w):
            return torch.zeros(2)

    model = Model()

    tree = record(model, lambda: model(model.lin.weight), iterations=1)

    assert not tree.weight_accesses.get("lin.weight")


class _Batch(UserDict):
    """UserDict mapping like transformers' ``BatchFeature``: pytree leaves it unflattened."""


def test_bytes_count_tensors_inside_a_userdict():
    class Model(nn.Module):
        def forward(self, batch):
            return _Batch({"x": batch["x"], "fresh": torch.zeros(4, 8)})

    model = Model()
    a, b, c = torch.randn(4, 8), torch.randn(2, 8), torch.randn(8)

    def call():
        return model(_Batch({"x": a, "y": b, "nested": [c]}))

    tree = record(model, call, iterations=1)

    node = tree.nodes[("root.forward",)]
    in_bytes = a.numel() * a.element_size() + b.numel() * b.element_size() + c.numel() * c.element_size()
    out_bytes = 2 * a.numel() * a.element_size()  # x plus a fresh same-shaped tensor
    assert node.bytes_in == in_bytes
    assert node.bytes_out == out_bytes


def test_bytes_count_a_tensor_in_a_userdict_and_an_arg_once():
    class Model(nn.Module):
        def forward(self, x, batch=None):
            return x

    model = Model()
    t = torch.randn(4, 8)

    tree = record(model, lambda: model(t, batch=_Batch({"again": t})), iterations=1)

    assert tree.nodes[("root.forward",)].bytes_in == t.numel() * t.element_size()


def test_bytes_count_tensors_inside_a_batch_feature():
    from transformers.feature_extraction_utils import BatchFeature

    class Model(nn.Module):
        def forward(self, features):
            return BatchFeature({"x": features["x"]})

    model = Model()
    a, b, c = torch.randn(4, 8), torch.randn(2, 8), torch.randn(8)

    tree = record(model, lambda: model(BatchFeature({"x": a, "y": b, "nested": [c]})), iterations=1)

    node = tree.nodes[("root.forward",)]
    in_bytes = a.numel() * a.element_size() + b.numel() * b.element_size() + c.numel() * c.element_size()
    assert node.bytes_in == in_bytes
    assert node.bytes_out == a.numel() * a.element_size()


def test_bytes_count_a_userdict_that_contains_itself():
    class Model(nn.Module):
        def forward(self, batch):
            return batch["x"]

    model = Model()
    t = torch.randn(4, 8)
    batch = _Batch({"x": t})
    batch["self"] = batch

    tree = record(model, lambda: model(batch), iterations=1)

    node = tree.nodes[("root.forward",)]
    assert node.bytes_in == t.numel() * t.element_size()
    assert node.bytes_out == t.numel() * t.element_size()


@pytest.mark.parametrize("options", [{}, {"frozen": True}, {"slots": True}], ids=["plain", "frozen", "slots"])
def test_bytes_count_tensors_inside_a_dataclass(options):
    @dataclasses.dataclass(**options)
    class Packed:
        x: torch.Tensor
        nested: list
        label: str = "packed"

    class Model(nn.Module):
        def forward(self, packed):
            return Packed(x=packed.x, nested=[torch.zeros(8)])

    model = Model()
    a, b = torch.randn(4, 8), torch.randn(2, 8)

    tree = record(model, lambda: model(Packed(x=a, nested=[b])), iterations=1)

    node = tree.nodes[("root.forward",)]
    assert node.bytes_in == a.numel() * a.element_size() + b.numel() * b.element_size()
    assert node.bytes_out == a.numel() * a.element_size() + 8 * 4


def test_bytes_count_a_dataclass_that_contains_itself():
    @dataclasses.dataclass
    class Node:
        x: torch.Tensor
        parent: object = None

    class Model(nn.Module):
        def forward(self, node):
            return node.x

    model = Model()
    t = torch.randn(4, 8)
    node = Node(t)
    node.parent = node

    tree = record(model, lambda: model(node), iterations=1)

    assert tree.nodes[("root.forward",)].bytes_in == t.numel() * t.element_size()


def test_bytes_skip_a_dataclass_field_without_a_value():
    @dataclasses.dataclass(slots=True)
    class Partial:
        x: torch.Tensor
        later: torch.Tensor = dataclasses.field(init=False)

    class Model(nn.Module):
        def forward(self, partial):
            return partial.x

    model = Model()
    t = torch.randn(4, 8)

    tree = record(model, lambda: model(Partial(t)), iterations=1)

    assert tree.nodes[("root.forward",)].bytes_in == t.numel() * t.element_size()


def test_bytes_walk_a_model_output_as_before():
    from transformers.modeling_outputs import BaseModelOutput

    class Model(nn.Module):
        def forward(self, x):
            return BaseModelOutput(last_hidden_state=x, hidden_states=(x, x[:2] * 0))

    model = Model()
    x = torch.randn(4, 8)

    tree = record(model, lambda: model(x), iterations=1)

    assert tree.nodes[("root.forward",)].bytes_out == x.numel() * x.element_size() + 2 * 8 * 4


def test_bytes_do_not_read_a_mapping_that_is_not_a_userdict():
    reads: list[str] = []

    class LazyMap(Mapping):
        def __getitem__(self, key):
            reads.append("__getitem__")
            raise KeyError(key)

        def __iter__(self):
            reads.append("__iter__")
            return iter(())

        def __len__(self):
            reads.append("__len__")
            return 0

    class Model(nn.Module):
        def forward(self, x, extra=None):
            return x

    model = Model()
    x = torch.randn(4, 8)

    tree = record(model, lambda: model(x, extra=LazyMap()), iterations=1)

    assert tree.nodes[("root.forward",)].bytes_in == x.numel() * x.element_size()
    assert reads == []


def _t(arg: Leaf) -> TensorLeaf:
    assert isinstance(arg, TensorLeaf)
    return arg


def test_signatures_name_inputs_after_the_bound_parameters():
    class Model(nn.Module):
        def forward(self, x, extra=None):
            return x

    model = Model()
    t, u = torch.randn(4, 8), torch.randn(2)

    tree = record(model, lambda: model(t, extra=u), iterations=1)

    entry = tree.nodes[("root.forward",)].signatures[0]
    assert set(entry.inputs) == {"x", "extra"}
    assert entry.outputs is not None
    assert set(entry.outputs) == {"output"}


def test_signatures_name_positional_and_keyword_arguments_the_same():
    class Model(nn.Module):
        def forward(self, x):
            return x

    model = Model()
    t = torch.randn(4, 8)

    tree = record(model, lambda: (model(t), model(x=t)), iterations=1)

    signature = tree.nodes[("root.forward",)].signatures[0]
    assert set(signature.inputs) == {"x"}


def test_signatures_of_a_kwargs_only_call():
    class Model(nn.Module):
        def forward(self, *, x):
            return x

    model = Model()
    t = torch.randn(4, 8)

    tree = record(model, lambda: model(x=t), iterations=1)

    assert set(tree.nodes[("root.forward",)].signatures[0].inputs) == {"x"}


def test_signatures_name_varargs_and_kwargs():
    class Model(nn.Module):
        def forward(self, *args, **kwargs):
            return args[0]

    model = Model()
    t, u, v = torch.randn(4, 8), torch.randn(2), torch.randn(1)

    tree = record(model, lambda: model(t, u, w=v), iterations=1)

    assert set(tree.nodes[("root.forward",)].signatures[0].inputs) == {
        "args[0]",
        "args[1]",
        "kwargs['w']",
    }


def test_signatures_fall_back_to_arg_positions_when_the_signature_cannot_be_bound():
    model = nn.Linear(2, 2)
    t = torch.randn(1, 2)

    recorder = CallRecorder()
    with recorder:
        recorder.begin_iteration()
        frame = recorder.begin_call("root.forward", model, "missing", (t,))
        recorder.end_call(frame, t)
        recorder.end_iteration()

    entry = recorder.result().nodes[("root.forward",)].signatures[0]
    assert set(entry.inputs) == {"args[0]"}


def test_signatures_of_a_plain_object_method():
    class Pipe:
        def run(self, x):
            return x

    pipe = Pipe()
    t = torch.randn(4, 8)

    tree = record(pipe, lambda: pipe.run(t), iterations=1)

    entry = tree.nodes[("root.run",)].signatures[0]
    assert set(entry.inputs) == {"x"}


def test_signatures_split_calls_with_a_changing_dimension():
    class Model(nn.Module):
        def forward(self, x):
            return x

    model = Model()
    a, b = torch.randn(1, 4, 8), torch.randn(1, 5, 8)

    tree = record(model, lambda: (model(a), model(b)), iterations=1)

    node = tree.nodes[("root.forward",)]
    assert [sig.call_indices for sig in node.signatures] == [[0], [1]]
    assert node.shapes("x") == [(1, 4, 8), (1, 5, 8)]


def test_signatures_reuse_the_argument_when_shapes_do_not_change():
    class Model(nn.Module):
        def forward(self, x):
            return x

    model = Model()

    # identical calls intern into one signature (fresh tensors: reusing x would give the later calls a source)
    tree = record(
        model, lambda: (model(torch.randn(4, 8)), model(torch.randn(4, 8)), model(torch.randn(4, 8))), iterations=1
    )
    node = tree.nodes[("root.forward",)]
    assert [sig.call_indices for sig in node.signatures] == [[0, 1, 2]]
    arg = tree.nodes[("root.forward",)].signatures[0].inputs["x"]
    assert isinstance(arg, TensorLeaf) and arg.shape == (4, 8)


def test_a_rank_change_at_a_path_is_a_separate_signature():
    class Model(nn.Module):
        def forward(self, x):
            return x

    model = Model()

    tree = record(
        model,
        lambda: (model(torch.randn(4, 8)), model(torch.randn(4, 8, 2)), model(torch.randn(4, 8, 3))),
        iterations=1,
    )

    node = tree.nodes[("root.forward",)]
    assert [sig.call_indices for sig in node.signatures] == [[0], [1], [2]]
    assert node.shapes("x") == [(4, 8), (4, 8, 2), (4, 8, 3)]


def test_a_rank_change_in_the_output_is_a_separate_signature():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.batched = True

        def forward(self, x):
            self.batched = not self.batched
            return x[None] if self.batched else x

    model = Model()

    tree = record(model, lambda: (model(torch.randn(4, 8)), model(torch.randn(4, 8))), iterations=1)

    node = tree.nodes[("root.forward",)]
    assert [sig.call_indices for sig in node.signatures] == [[0], [1]]
    assert node.shapes("output", "outputs") == [(4, 8), (1, 4, 8)]


def test_a_dtype_change_at_a_path_is_a_separate_signature():
    class Model(nn.Module):
        def forward(self, x):
            return x

    model = Model()

    tree = record(
        model,
        lambda: (model(torch.randn(4, 8)), model(torch.randn(4, 8, dtype=torch.bfloat16))),
        iterations=1,
    )

    node = tree.nodes[("root.forward",)]
    assert [sig.call_indices for sig in node.signatures] == [[0], [1]]
    assert _t(node.signatures[0].inputs["x"]).dtype == torch.float32
    assert _t(node.signatures[1].inputs["x"]).dtype == torch.bfloat16


def test_signatures_split_prefill_and_decode_calls():
    class Model(nn.Module):
        def forward(self, x, cache_position=None):
            return x

    model = Model()

    tree = record(
        model,
        lambda: (
            model(torch.randn(1, 4, 8), cache_position=torch.arange(4)),  # prefill-like
            model(torch.randn(1, 1, 8), cache_position=torch.tensor([4])),  # decode-like
        ),
        iterations=1,
    )

    node = tree.nodes[("root.forward",)]
    assert node.shapes("x") == [(1, 4, 8), (1, 1, 8)]
    assert node.shapes("cache_position") == [(4,), (1,)]


def test_signatures_walk_nested_containers_a_model_output_and_a_batch_feature():
    from transformers.feature_extraction_utils import BatchFeature
    from transformers.modeling_outputs import BaseModelOutputWithPast

    class Model(nn.Module):
        def forward(self, x, features):
            return BaseModelOutputWithPast(last_hidden_state=x)

    model = Model()
    t, u, v, w = torch.randn(4, 8), torch.randn(2), torch.randn(1), torch.randn(3)

    call = lambda: model([t, (u, {"k": v})], features=BatchFeature({"f": w}))  # noqa: E731
    tree = record(model, call, iterations=1)

    entry = tree.nodes[("root.forward",)].signatures[0]
    assert set(entry.inputs) == {"x[0]", "x[1][0]", "x[1][1]['k']", "features['f']"}
    outputs = entry.outputs
    assert outputs is not None
    assert set(outputs) == {
        "output['last_hidden_state'][0]",
        "output['last_hidden_state'][1][0]",
        "output['last_hidden_state'][1][1]['k']",
    }


def test_signatures_list_tensors_in_container_order():
    from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

    class Model(nn.Module):
        def forward(self, x):
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=x["b"], hidden_states=(x["a"][0], x["a"][1][0])
            )

    model = Model()
    tensors = [torch.randn(1), torch.randn(2), torch.randn(3), torch.randn(4)]
    nested = {"a": [tensors[0], (tensors[1], tensors[2])], "b": tensors[3]}

    tree = record(model, lambda: model(nested), iterations=1)

    entry = tree.nodes[("root.forward",)].signatures[0]
    outputs = entry.outputs
    assert outputs is not None
    assert list(entry.inputs) == ["x['a'][0]", "x['a'][1][0]", "x['a'][1][1]", "x['b']"]
    assert list(outputs) == [
        "output['last_hidden_state']",
        "output['hidden_states'][0]",
        "output['hidden_states'][1]",
    ]
    entries = tree.to_dict()["nodes"][0]
    exported = entries["signatures"][0]
    assert list(exported["inputs"]) == list(entry.inputs)
    assert list(exported["outputs"]) == list(outputs)


def test_bytes_skip_and_signatures_list_a_static_cache_by_type_name():
    from transformers import LlamaConfig
    from transformers.cache_utils import StaticCache

    class Model(nn.Module):
        def forward(self, x, past_key_values):
            return x

    config = LlamaConfig(
        hidden_size=8,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    cache = StaticCache(config=config, max_cache_len=16)
    for layer in range(2):
        cache.update(torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4), layer)
    model = Model()
    x = torch.randn(4, 8)

    tree = record(model, lambda: model(x, cache), iterations=1)

    node = tree.nodes[("root.forward",)]
    # a StaticCache is a leaf to the walk: passed by reference, not as data crossing the call
    assert node.signatures[0].inputs["past_key_values"].type == "transformers.cache_utils.StaticCache"
    assert node.bytes_in == x.numel() * x.element_size()


def test_bytes_skip_and_signatures_list_a_dynamic_cache_by_type_name():
    from transformers.cache_utils import DynamicCache

    class Model(nn.Module):
        def forward(self, x, past_key_values):
            return x

    cache = DynamicCache()
    cache.update(torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4), 0)
    model = Model()
    x = torch.randn(4, 8)

    tree = record(model, lambda: model(x, cache), iterations=1)

    node = tree.nodes[("root.forward",)]
    assert node.signatures[0].inputs["past_key_values"].type == "transformers.cache_utils.DynamicCache"
    assert node.bytes_in == x.numel() * x.element_size()


def test_signatures_list_a_dict_subclass_by_type_name_and_walk_a_dataclass_into_its_fields():
    class Box(dict):
        pass

    import dataclasses

    @dataclasses.dataclass
    class Pack:
        x: torch.Tensor
        y: torch.Tensor

    class Model(nn.Module):
        def forward(self, box, pack):
            return box["a"]

    model = Model()
    a, b, c = torch.randn(4, 8), torch.randn(2), torch.randn(1)

    tree = record(model, lambda: model(Box(a=a), Pack(x=b, y=c)), iterations=1)

    node = tree.nodes[("root.forward",)]
    # a dict subclass is an opaque leaf to the walk; a dataclass is walked into its fields
    assert node.bytes_in == b.numel() * b.element_size() + c.numel() * c.element_size()
    inputs = node.signatures[0].inputs
    assert inputs["box"].type == f"{Box.__module__}.{Box.__qualname__}"
    assert [(name, _t(inputs[name]).shape) for name in ("pack.x", "pack.y")] == [("pack.x", (2,)), ("pack.y", (1,))]
    assert "pack" not in inputs
    outputs = node.signatures[0].outputs
    assert outputs is not None
    assert set(outputs) == {"output"}
    assert node.bytes_out == a.numel() * a.element_size()


def test_bytes_and_shapes_count_a_tensor_inside_a_deque():
    from collections import deque

    class Model(nn.Module):
        def forward(self, x, items):
            return x

    model = Model()
    x, t = torch.randn(4, 8), torch.randn(2)

    tree = record(model, lambda: model(x, deque([t])), iterations=1)

    node = tree.nodes[("root.forward",)]
    # pytree flattens a deque, so it is a container to bytes and shapes alike
    assert node.bytes_in == (x.numel() + t.numel()) * x.element_size()
    assert set(node.signatures[0].inputs) == {"x", "items[0]"}


def test_result_snapshots_the_signature():
    class Model(nn.Module):
        def forward(self, x):
            return x

    model = Model()

    recorder = CallRecorder()
    with instrumented(recorder, model), recorder:
        recorder.begin_iteration()
        model(torch.randn(1, 2))
        recorder.end_iteration()
        tree = recorder.result()
        recorder.begin_iteration()
        model(torch.randn(1, 3))  # adds a second signature to the live recorder only
        model(torch.randn(1, 2))  # adds a call to the first signature in the live recorder only
        recorder.end_iteration()

    snapshot = tree.nodes[("root.forward",)]
    assert [sig.call_indices for sig in snapshot.signatures] == [[0]]
    assert _t(snapshot.signatures[0].inputs["x"]).shape == (1, 2)
    live = recorder.result().nodes[("root.forward",)]
    assert [sig.call_indices for sig in live.signatures] == [[0, 2], [1]]


def test_result_snapshots_the_signature_mappings_too():
    class Model(nn.Module):
        def forward(self, x):
            return x

    model = Model()

    recorder = CallRecorder()
    with instrumented(recorder, model), recorder:
        recorder.begin_iteration()
        model(torch.randn(1, 2))
        recorder.end_iteration()

    first = recorder.result().nodes[("root.forward",)].signatures[0]
    assert first.outputs is not None
    first.inputs.pop("x")
    first.outputs.pop("output")

    with instrumented(recorder, model), recorder:
        recorder.begin_iteration()
        model(torch.randn(1, 2))  # same signature
        recorder.end_iteration()

    second = recorder.result().nodes[("root.forward",)].signatures[0]
    assert second.outputs is not None
    assert set(second.inputs) == {"x"}
    assert set(second.outputs) == {"output"}
    assert second.call_indices == [0, 1]


def test_signatures_record_objects_by_type_name_only():
    reads = []

    class Guarded:
        def __getattribute__(self, name):
            if name != "__class__":
                reads.append(name)
            return object.__getattribute__(self, name)

    class Model(nn.Module):
        def forward(self, x, obj):
            return x

    model = Model()

    tree = record(model, lambda: model(torch.randn(2), Guarded()), iterations=1)

    leaf = tree.nodes[("root.forward",)].signatures[0].inputs["obj"]
    assert leaf.type == f"{Guarded.__module__}.{Guarded.__qualname__}"
    assert not isinstance(leaf, TensorLeaf)
    assert reads == []


def test_a_tensor_and_an_object_at_the_same_path_give_separate_signatures():
    class Marker:
        pass

    class Model(nn.Module):
        def forward(self, x):
            return None

    model = Model()

    tree = record(model, lambda: (model(torch.randn(2)), model(Marker())), iterations=1)

    signatures = tree.nodes[("root.forward",)].signatures
    assert [sig.call_indices for sig in signatures] == [[0], [1]]
    assert isinstance(signatures[0].inputs["x"], TensorLeaf)
    assert type(signatures[1].inputs["x"]) is Leaf


def test_argument_names_are_the_signature_keys():
    class Model(nn.Module):
        def forward(self, inputs):
            return {"logits": inputs["input_ids"].float()}

    model = Model()

    tree = record(model, lambda: model({"input_ids": torch.zeros(1, 3, dtype=torch.long)}), iterations=1)

    signature = tree.nodes[("root.forward",)].signatures[0]
    assert list(signature.inputs) == ["inputs['input_ids']"]
    assert list(signature.outputs or {}) == ["output['logits']"]
    for arguments in (signature.inputs, signature.outputs or {}):
        assert all(arg.name == name for name, arg in arguments.items())


def test_signatures_do_not_record_primitives():
    import enum

    class Mode(enum.Enum):
        FAST = "fast"

    class Model(nn.Module):
        def forward(self, x, flag, n, s, d, e, m):
            return x

    model = Model()
    x = torch.randn(4, 8)

    tree = record(
        model,
        lambda: (
            model(x, True, 3, "x", torch.bfloat16, Mode.FAST, None),
            model(x, False, 9, "other", torch.float32, Mode.FAST, 0),
        ),
        iterations=1,
    )

    signature = tree.nodes[("root.forward",)].signatures[0]
    assert set(signature.inputs) == {"x"}


def test_signature_counts_the_calls_each_path_was_passed():
    class Model(nn.Module):
        def forward(self, x):
            return x

    model = Model()

    tree = record(model, lambda: (model(object()), model(None)), iterations=1)

    node = tree.nodes[("root.forward",)]
    assert node.calls == 2
    assert node.calls_passing("x") == [0]


def test_signatures_record_a_non_tensor_result():
    class Model(nn.Module):
        def forward(self, x):
            return x, object(), None

    model = Model()

    tree = record(model, lambda: model(torch.randn(2, 3)), iterations=1)

    outputs = tree.nodes[("root.forward",)].signatures[0].outputs
    assert outputs is not None
    assert isinstance(outputs["output[0]"], TensorLeaf)
    assert outputs["output[1]"].type == "object"
    assert "output[2]" not in outputs


def test_bytes_do_not_count_objects_or_primitives():
    class Model(nn.Module):
        def forward(self, x, flag, note, obj):
            return x

    model = Model()
    x = torch.randn(4, 8)

    tree = record(model, lambda: model(x, True, "note", object()), iterations=1)

    node = tree.nodes[("root.forward",)]
    assert node.bytes_in == x.numel() * x.element_size()
    assert set(node.signatures[0].inputs) == {"x", "obj"}


def test_signatures_walk_terminates_on_a_self_referential_list():
    class Model(nn.Module):
        def forward(self, x):
            return x

    model = Model()
    t = torch.randn(4, 8)
    items: list = [t]
    items.append(items)

    tree = record(model, lambda: model(items), iterations=1)

    node = tree.nodes[("root.forward",)]
    assert node.bytes_in == t.numel() * t.element_size()
    assert list(node.signatures[0].inputs) == ["x[0]"]


def test_signatures_walk_terminates_on_a_self_referential_dict():
    class Model(nn.Module):
        def forward(self, x):
            return x

    model = Model()
    t = torch.randn(4, 8)
    box: dict = {"t": t}
    box["self"] = box

    tree = record(model, lambda: model(box), iterations=1)

    node = tree.nodes[("root.forward",)]
    assert node.bytes_in == t.numel() * t.element_size()
    assert list(node.signatures[0].inputs) == ["x['t']"]


def test_signatures_walk_terminates_on_an_indirect_cycle():
    class Model(nn.Module):
        def forward(self, x):
            return x

    model = Model()
    t = torch.randn(4, 8)
    a: dict = {"t": t}
    b = [a]
    a["b"] = b

    tree = record(model, lambda: model(a), iterations=1)

    node = tree.nodes[("root.forward",)]
    assert node.bytes_in == t.numel() * t.element_size()
    assert list(node.signatures[0].inputs) == ["x['t']"]


def test_a_container_shared_between_arguments_is_walked_under_its_first_path():
    class Model(nn.Module):
        def forward(self, x, y):
            return x

    model = Model()
    t = torch.randn(4, 8)
    shared = (t, t)

    tree = record(model, lambda: model(shared, shared), iterations=1)

    node = tree.nodes[("root.forward",)]
    assert node.bytes_in == t.numel() * t.element_size()
    assert list(node.signatures[0].inputs) == ["x[0]", "x[1]"]


def test_signatures_record_a_parameters_class():
    class Model(nn.Module):
        def forward(self, x):
            return x

    model = Model()
    weight = nn.Parameter(torch.randn(4, 8))

    tree = record(model, lambda: model(weight), iterations=1)

    arg = tree.nodes[("root.forward",)].signatures[0].inputs["x"]
    assert arg.type == "torch.nn.parameter.Parameter"
    assert isinstance(arg, TensorLeaf) and arg.shape == (4, 8)


def test_a_tensor_and_a_parameter_at_one_path_are_two_signatures():
    class Model(nn.Module):
        def forward(self, x):
            return x

    model = Model()

    tree = record(model, lambda: (model(torch.randn(4, 8)), model(nn.Parameter(torch.randn(4, 8)))), iterations=1)

    node = tree.nodes[("root.forward",)]
    assert [sig.call_indices for sig in node.signatures] == [[0], [1]]
    assert node.signatures[0].inputs["x"].type == "torch.Tensor"
    assert node.signatures[1].inputs["x"].type == "torch.nn.parameter.Parameter"


def _x(signature: CallSignature) -> TensorLeaf:
    arg = signature.inputs["x"]
    assert isinstance(arg, TensorLeaf)
    return arg


def test_calls_passing_and_shapes_cover_inputs_and_outputs():
    class Model(nn.Module):
        def forward(self, x):
            return x

    model = Model()

    tree = record(
        model,
        lambda: (model(torch.randn(4, 8)), model(torch.randn(2, 8)), model(torch.randn(4, 8))),
        iterations=1,
    )

    node = tree.nodes[("root.forward",)]
    assert node.calls_passing("x") == [0, 1, 2]
    assert node.shapes("x") == [(4, 8), (2, 8)]
    assert node.calls_passing("output", "outputs") == [0, 1, 2]
    assert node.shapes("output", "outputs") == [(4, 8), (2, 8)]
    assert node.calls_passing("missing") == []


def test_calls_passing_skips_signatures_whose_calls_raised():
    class Model(nn.Module):
        def forward(self, x):
            if x.ndim > 2:
                raise ValueError("too many dims")
            return x

    model = Model()

    recorder = CallRecorder()
    with instrumented(recorder, model), recorder:
        recorder.begin_iteration()
        model(torch.randn(4, 8))
        with pytest.raises(ValueError):
            model(torch.randn(1, 4, 8))
        recorder.end_iteration()

    node = recorder.result().nodes[("root.forward",)]
    assert node.calls_passing("x") == [0, 1]
    assert node.calls_passing("output", "outputs") == [0]  # the raised call passed no result


def test_iteration_starts_count_calls_before_each_iteration():
    class Model(nn.Module):
        def forward(self, x):
            return x

    model = Model()
    x = torch.randn(4, 8)

    recorder = CallRecorder()
    with instrumented(recorder, model), recorder:
        recorder.begin_iteration()
        model(x)
        model(x)
        recorder.end_iteration()
        recorder.begin_iteration()  # no call in this iteration
        recorder.end_iteration()
        recorder.begin_iteration()
        model(x)
        recorder.end_iteration()

    tree = recorder.result()
    assert tree.iteration_starts()[("root.forward",)] == [0, 2, 2]


def test_call_indices_continue_across_iterations():
    class Model(nn.Module):
        def forward(self, x):
            return x

    model = Model()
    x = torch.randn(4, 8)

    recorder = CallRecorder()
    with instrumented(recorder, model), recorder:
        recorder.begin_iteration()
        model(x)
        recorder.end_iteration()
        recorder.begin_iteration()
        model(torch.randn(2, 8))
        model(x)
        recorder.end_iteration()

    node = recorder.result().nodes[("root.forward",)]
    assert [sig.call_indices for sig in node.signatures] == [[0, 2], [1]]
    assert node.calls == 3


def test_signatures_fall_back_to_arg_positions_when_the_call_does_not_bind():
    class Model(nn.Module):
        def forward(self, x):
            return x

    model = Model()
    t, u = torch.randn(1, 2), torch.randn(1, 3)

    recorder = CallRecorder()
    with recorder:
        recorder.begin_iteration()
        frame = recorder.begin_call("root.forward", model, "forward", (t, u))
        recorder.end_call(frame, t)
        recorder.end_iteration()

    node = recorder.result().nodes[("root.forward",)]
    assert set(node.signatures[0].inputs) == {"args[0]", "args[1]"}


def test_names_use_positional_steps_for_a_non_string_pytree_context():
    class Pair:
        def __init__(self, a, b):
            self.a = a
            self.b = b

    torch.utils._pytree.register_pytree_node(
        Pair,
        lambda pair: ([pair.a, pair.b], [0, 1]),
        lambda values, context: Pair(*values),
    )
    try:

        class Model(nn.Module):
            def forward(self, x):
                return x

        model = Model()

        tree = record(model, lambda: model(Pair(torch.randn(2), torch.randn(3))), iterations=1)

        node = tree.nodes[("root.forward",)]
        assert list(node.signatures[0].inputs) == ["x[0]", "x[1]"]
    finally:
        torch.utils._pytree._deregister_pytree_node(Pair)


def test_names_use_positional_steps_for_a_repeated_pytree_context_name():
    class Pair:
        def __init__(self, a, b):
            self.a = a
            self.b = b

    torch.utils._pytree.register_pytree_node(
        Pair,
        lambda pair: ([pair.a, pair.b], ["batch", "batch"]),
        lambda values, context: Pair(*values),
    )
    try:

        class Model(nn.Module):
            def forward(self, x):
                return x

        model = Model()

        tree = record(model, lambda: model(Pair(torch.randn(2), torch.randn(3))), iterations=1)

        node = tree.nodes[("root.forward",)]
        assert list(node.signatures[0].inputs) == ["x[0]", "x[1]"]
    finally:
        torch.utils._pytree._deregister_pytree_node(Pair)


def test_signatures_fall_back_without_losing_positional_arguments_to_a_kwarg_named_args():
    model = nn.Linear(2, 2)
    t, u = torch.randn(1, 2), torch.randn(1, 3)

    recorder = CallRecorder()
    with recorder:
        recorder.begin_iteration()
        frame = recorder.begin_call("root.forward", model, "missing", (t,), {"args": u})
        recorder.end_call(frame, t)
        recorder.end_iteration()

    node = recorder.result().nodes[("root.forward",)]
    assert set(node.signatures[0].inputs) == {"args[0]", "kwargs['args']"}


def test_signatures_bind_each_instance_with_its_own_method():
    class Model(nn.Module):
        def forward(self, x):
            return x

    plain, patched = Model(), Model()
    # an instance-level replacement with another parameter name
    patched.forward = lambda y: y  # type: ignore[assignment]
    t = torch.randn(1, 2)

    recorder = CallRecorder()
    with recorder:
        recorder.begin_iteration()
        for label, model in (("root.plain.forward", plain), ("root.patched.forward", patched)):
            frame = recorder.begin_call(label, model, "forward", (t,))
            recorder.end_call(frame, t)
        recorder.end_iteration()

    nodes = recorder.result().nodes
    assert set(nodes[("root.plain.forward",)].signatures[0].inputs) == {"x"}
    assert set(nodes[("root.patched.forward",)].signatures[0].inputs) == {"y"}


def test_recording_a_call_dispatches_no_torch_function_for_its_leaves(mocker):
    class Model(nn.Module):
        def forward(self, x, items):
            return x

    model = Model()
    t, u, v = torch.randn(4, 8), torch.randn(2), torch.randn(3)

    spy = mocker.spy(
        __import__("aitune.torch.call_tree.weights.accesses", fromlist=["WeightAccessMode"]).WeightAccessMode,
        "__torch_function__",
    )
    tree = record(model, lambda: model(t, items=[u, {"k": v}]), iterations=1)

    spy.assert_not_called()
    assert tree.nodes[("root.forward",)].bytes_in > 0
