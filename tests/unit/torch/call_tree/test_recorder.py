# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import UserDict
from collections.abc import Mapping

import pytest
import torch
from torch import nn

from aitune.torch.call_tree.callers import OUTSIDE_ANY_CALL
from aitune.torch.call_tree.methods import is_transparent_call, is_transparent_container, user_defined_methods
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
    assert [n.key[-2] for n in llm_nodes] == ["root.prefill", "root.generate"]
    assert tree.nodes_of(model.llm) == llm_nodes


def test_nodes_of_finds_a_modules_nodes_by_identity_and_method():
    model = VlaToy(steps=2)
    tokens, image = vla_inputs()

    tree = record(model, lambda: model(tokens, image))

    assert [n.path for n in tree.nodes_of(model.dit)] == ["forward > dit_forward > dit.forward"]
    assert [n.path for n in tree.nodes_of(model, "dit_forward")] == ["forward > dit_forward"]
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

    assert recorder._ms[("root.forward",)] == 2.0
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
            return BaseModelOutputWithPast(last_hidden_state=x, past_key_values=DynamicCache())

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
