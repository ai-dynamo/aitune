# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools

import pytest
import torch
from torch import nn

from aitune.torch.call_tree.call_tree import CallTree
from aitune.torch.call_tree.nodes import CallNode
from aitune.torch.call_tree.recording import CallRecorder, StructureRecorder, TimingRecorder
from tests.toy_models.call_tree_models import TwoRoleToy
from tests.unit.torch.call_tree.helpers import instrumented


def _tree_of(calls_per_sample: list[list[tuple[str, ...]]]) -> CallTree:
    """A call tree whose nodes are the paths ``calls_per_sample`` lists, with those counts per sample."""
    nodes = {}
    for paths in calls_per_sample:
        for path in paths:
            if path not in nodes:
                nodes[path] = CallNode(path=path, label=path[-1], owner_class="M", method="forward")
                nodes[path].calls_per_sample = [0] * len(calls_per_sample)
    for i, paths in enumerate(calls_per_sample):
        for path in paths:
            nodes[path].calls_per_sample[i] += 1
    return CallTree(nodes=nodes)


def _boom() -> None:
    raise RuntimeError("boom")


def test_every_sample_replays_once_per_depth_its_calls_reach():
    paths: list[tuple[str, ...]] = [("a",), ("a", "b"), ("a", "b", "c")]
    tree = _tree_of([paths for _ in range(2)])

    assert TimingRecorder(tree).depths_per_sample() == [[1, 2, 3], [1, 2, 3]]


def test_a_depth_only_some_samples_reach_is_planned_only_for_them():
    root, child = ("a",), ("a", "b")
    tree = _tree_of([[root, child], [root]])

    assert TimingRecorder(tree).depths_per_sample() == [[1, 2], [1]]


def test_a_sample_without_calls_gets_no_replays():
    tree = _tree_of([[("a",)], []])

    assert TimingRecorder(tree).depths_per_sample() == [[1], []]


def test_calls_inside_a_compiled_part_are_not_planned():
    """A node that only runs inside a compiled part is left out of the plan entirely."""
    root, inner = ("a",), ("a", "b")
    tree = _tree_of([[root, inner]] * 2)
    tree.nodes[inner].inside_compiled_part = True

    assert TimingRecorder(tree).depths_per_sample() == [[1], [1]]


def test_a_sample_with_only_inside_compiled_calls_gets_no_replays():
    root, inner = ("a",), ("a", "b")
    tree = _tree_of([[root], [inner]])
    tree.nodes[inner].inside_compiled_part = True

    assert TimingRecorder(tree).depths_per_sample() == [[1], []]


def test_max_depth_limits_the_plan_to_shallower_depths():
    paths: list[tuple[str, ...]] = [("a",), ("a", "b"), ("a", "b", "c")]
    tree = _tree_of([paths for _ in range(2)])

    assert TimingRecorder(tree, max_depth=2).depths_per_sample() == [[1, 2], [1, 2]]


def test_a_sample_with_only_calls_deeper_than_max_depth_gets_no_replays():
    root, inner = ("a",), ("a", "b")
    tree = _tree_of([[root], [inner]])

    assert TimingRecorder(tree, max_depth=1).depths_per_sample() == [[1], []]


def test_calls_inside_a_timed_call_are_skipped(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=False)
    tree = _tree_with_a_child()
    model = nn.Linear(2, 2)
    ran = []
    stack_during_root = []

    with TimingRecorder(tree) as timer:
        with timer.time_replay(1):  # this replay times depth 1: the root

            def child_forward():
                ran.append(True)

            def root_forward():
                stack_during_root.append(len(timer._stack))
                # a call inside a timed call is deeper than the timed depth: it runs but is not recorded
                timer.call(child_forward, (), {}, label="root.child.forward", obj=model, method="forward")

            timer.call(root_forward, (), {}, label="root.forward", obj=model, method="forward")

    child_path = ("root.forward", "root.child.forward")
    assert ran and stack_during_root == [1]
    assert timer._stack == []
    assert tree.nodes[("root.forward",)].total_time_ms is not None
    assert tree.nodes[child_path].total_time_ms is None


def test_the_structure_recorder_does_not_skip_calls():
    assert not StructureRecorder()._skips_calls()


def _perf_reads(durations_ms: list[float]) -> list[float]:
    """``perf_counter`` reads (seconds) that make consecutive calls take ``durations_ms`` each: 0, then the end."""
    return [t for d in durations_ms for t in (0.0, d / 1000)]


def test_the_total_counts_a_systematically_slow_call_of_each_sample(mocker):
    """A slow first call per sample, like a prefill before decode steps, must reach the total."""
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=False)
    mocker.patch(
        "aitune.torch.call_tree.recording.timing_recorder.time.perf_counter",
        side_effect=_perf_reads([4.0, 1.0, 1.0] * 3),  # calls of 4, 1 and 1 ms in each of 3 samples
    )
    a = ("a",)
    tree = _tree_of([[a, a, a]] * 3)
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        for _ in range(3):
            with timer.time_replay(1):
                for _ in range(3):
                    timer.end_call(timer.begin_call("a", model, "forward", ()))

    node = tree.nodes[a]
    # 18 ms over 9 calls; a per-call median would give 1 ms
    assert node.total_time_ms == pytest.approx(18.0)


def test_a_slow_sample_counts_in_the_total(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=False)
    mocker.patch(
        "aitune.torch.call_tree.recording.timing_recorder.time.perf_counter",
        side_effect=_perf_reads([1.0, 3.0, 1.0, 3.0, 18.0, 22.0]),  # 4 ms, 4 ms and 40 ms per sample
    )
    a = ("a",)
    tree = _tree_of([[a, a], [a, a], [a, a]])
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        for _ in range(3):
            with timer.time_replay(1):
                for _ in range(2):
                    timer.end_call(timer.begin_call("a", model, "forward", ()))

    node = tree.nodes[a]
    assert node.total_time_ms == pytest.approx(48.0)  # the measured sum: the slow sample counts


def test_time_ms_is_none_without_a_timing_pass():
    node = CallNode(path=("a",), label="a", owner_class="M", method="forward", calls_per_sample=[2])

    assert node.total_time_ms is None


def test_depths_per_sample_of_a_tree_with_no_timeable_nodes():
    assert TimingRecorder(CallTree()).depths_per_sample() == []
    node = CallNode(
        path=("a",),
        label="a",
        owner_class="M",
        method="forward",
        calls_per_sample=[1, 1],
        inside_compiled_part=True,
    )
    assert TimingRecorder(CallTree(nodes={node.path: node})).depths_per_sample() == []


def test_the_pass_writes_each_nodes_measured_time(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=False)
    # root calls take 2, 6 and 5 ms; the child calls 1, 3, 4 and 7 ms over the three samples
    mocker.patch(
        "aitune.torch.call_tree.recording.timing_recorder.time.perf_counter",
        side_effect=_perf_reads([2.0, 1.0, 6.0, 3.0, 4.0, 5.0, 7.0]),
    )
    root_path, child_path = ("root.forward",), ("root.forward", "root.child.forward")
    tree = _tree_of([[root_path, child_path], [root_path, child_path, child_path], [root_path, child_path]])
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        for i in range(len(tree.nodes[root_path].calls_per_sample)):
            with timer.time_replay(1):  # times the root; the deeper child calls are skipped
                timer.end_call(timer.begin_call("root.forward", model, "forward", ()))
            with timer.time_replay(2):  # times the children
                root = timer.begin_call("root.forward", model, "forward", ())
                for _ in range(tree.nodes[child_path].calls_per_sample[i]):
                    timer.end_call(timer.begin_call("root.child.forward", model, "forward", ()))
                timer.end_call(root)

    assert tree.nodes[root_path].total_time_ms == pytest.approx(13.0)
    assert tree.nodes[child_path].total_time_ms == pytest.approx(15.0)


def _recorded_tree(model, x):
    """The tree of one structure-pass sample of ``model(x)``."""
    recorder = StructureRecorder()
    with instrumented(recorder, model), recorder:
        with recorder.record_sample():
            model(x)
    return recorder.result()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA events require a GPU")
def test_time_replay_resolves_pending_events():
    model = nn.Linear(2, 2).cuda()
    x = torch.randn(2, 2, device="cuda")
    tree = _recorded_tree(model, x)

    with TimingRecorder(tree) as timer:
        with timer.time_replay(1):
            call = timer.begin_call("root.forward", model, "forward", ())
            model(x)
            timer.end_call(call)

            assert timer._events
        assert not timer._events
        assert timer._free_events  # resolved events are reused

    node = tree.nodes[("root.forward",)]
    assert node.total_time_ms is not None and node.total_time_ms > 0


def test_a_model_on_the_cpu_is_timed_with_perf_counter(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=True)
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_initialized", return_value=True)
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.memory_reserved", return_value=0)
    event = mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.Event")
    model = nn.Linear(2, 2)
    x = torch.randn(2, 2)
    tree = _recorded_tree(model, x)

    with TimingRecorder(tree) as timer:
        assert not timer._use_cuda
        with timer.time_replay(1):
            timer.end_call(timer.begin_call("root.forward", model, "forward", ()))

    event.assert_not_called()
    assert tree.nodes[("root.forward",)].total_time_ms is not None


def test_a_model_in_a_process_with_cuda_memory_is_timed_with_cuda_events(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=True)
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_initialized", return_value=True)
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.memory_reserved", return_value=1)
    model = nn.Linear(2, 2)
    x = torch.randn(2, 2)
    tree = _recorded_tree(model, x)

    with TimingRecorder(tree) as timer:
        assert timer._use_cuda


def test_cuda_memory_reserved_but_nothing_allocated_still_uses_cuda_events(mocker):
    """Freed CUDA tensors keep the allocator's memory reserved: earlier CUDA work still means CUDA events."""
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=True)
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_initialized", return_value=True)
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.memory_allocated", return_value=0)
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.memory_reserved", return_value=1)
    model = nn.Linear(2, 2)
    x = torch.randn(2, 2)
    tree = _recorded_tree(model, x)

    with TimingRecorder(tree) as timer:
        assert timer._use_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA events require a GPU")
def test_a_model_on_the_gpu_is_timed_with_cuda_events():
    model = nn.Linear(2, 2).cuda()
    tree = _recorded_tree(model, torch.randn(2, 2, device="cuda"))

    with TimingRecorder(tree) as timer:
        assert timer._use_cuda


def test_end_event_is_recorded_on_the_start_stream_when_the_call_switches_streams(mocker):
    """A call that switches the current stream still ends its timer on the stream it started on."""
    events = []

    class FakeEvent:
        def __init__(self, *args, **kwargs):
            events.append(self)
            self.stream = None
            self.synchronized = False

        def synchronize(self):
            self.synchronized = True

        def elapsed_time(self, end):
            return 2.0

    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.Event", FakeEvent)
    mocker.patch(
        "aitune.torch.call_tree.recording.timing_recorder._record",
        side_effect=lambda event, stream: setattr(event, "stream", stream),
    )
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.current_device", return_value=0)
    current = ["stream0"]
    mocker.patch.object(TimingRecorder, "_current_stream", lambda self, device: current[0])
    model = nn.Linear(2, 2)
    x = torch.randn(2, 2)
    tree = _recorded_tree(model, x)

    with TimingRecorder(tree) as timer:
        timer._use_cuda = True
        with timer.time_replay(1):
            call = timer.begin_call("root.forward", model, "forward", ())
            current[0] = "stream1"  # the call switched the current stream
            timer.end_call(call)

    start, end = events
    assert start.stream == "stream0"
    assert end.stream == "stream0"
    assert tree.nodes[("root.forward",)].total_time_ms == 2.0


def test_time_replay_synchronizes_every_end_event(mocker):
    events = []

    class FakeEvent:
        def __init__(self, *args, **kwargs):
            events.append(self)
            self.synchronized = False

        def synchronize(self):
            self.synchronized = True

        def elapsed_time(self, end):
            if not end.synchronized:
                raise RuntimeError("Both events must be completed")
            return 1.0

    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.Event", FakeEvent)
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder._record", side_effect=lambda event, stream: None)
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.current_device", return_value=0)
    mocker.patch.object(TimingRecorder, "_current_stream", lambda self, device: object())
    cuda_synchronize = mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.synchronize")
    model = nn.Linear(2, 2)
    x = torch.randn(2, 2)
    recorder = StructureRecorder()
    with instrumented(recorder, model), recorder:
        with recorder.record_sample():
            model(x)
            model(x)
    tree = recorder.result()  # the root has total_calls=2 in one sample

    with TimingRecorder(tree) as timer:
        timer._use_cuda = True
        with timer.time_replay(1):
            for _ in range(2):
                timer.end_call(timer.begin_call("root.forward", model, "forward", ()))

    node = tree.nodes[("root.forward",)]
    assert node.total_time_ms == 2.0  # the measured sum of the two 1.0 ms calls
    cuda_synchronize.assert_not_called()


def test_unresolved_cuda_times_all_reach_the_total(mocker):
    elapsed = iter([1.0, 1.0, 100.0])

    class FakeEvent:
        def __init__(self, *args, **kwargs):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, end):
            return next(elapsed)

    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.Event", FakeEvent)
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder._record", side_effect=lambda event, stream: None)
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.current_device", return_value=0)
    mocker.patch.object(TimingRecorder, "_current_stream", lambda self, device: object())
    a = ("a",)
    tree = _tree_of([[a], [a], [a]])
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        timer._use_cuda = True
        for _ in range(3):
            with timer.time_replay(1):
                timer.end_call(timer.begin_call("a", model, "forward", ()))
        # consecutive replays: all event pairs still land in the total

    node = tree.nodes[a]
    assert node.total_time_ms == pytest.approx(102.0)  # the measured sum


def test_resolved_events_are_reused_only_on_their_device(mocker):
    """The event pool is per device: a pair resolved on device 0 must not serve a call on device 1."""
    created = []

    class FakeEvent:
        def __init__(self, *args, **kwargs):
            created.append(self)

        def synchronize(self):
            pass

        def elapsed_time(self, end):
            return 1.0

    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.Event", FakeEvent)
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder._record", side_effect=lambda event, stream: None)
    device = [0]
    mocker.patch(
        "aitune.torch.call_tree.recording.timing_recorder.torch.cuda.current_device", side_effect=lambda: device[0]
    )
    mocker.patch.object(TimingRecorder, "_current_stream", lambda self, dev: f"stream{dev}")
    a = ("a",)
    tree = _tree_of([[a], [a]])
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        timer._use_cuda = True
        with timer.time_replay(1):
            timer.end_call(timer.begin_call("a", model, "forward", ()))
        assert timer._free_events[0] == created[:2]  # the resolved pair went back to device 0's pool

        device[0] = 1
        with timer.time_replay(1):
            timer.end_call(timer.begin_call("a", model, "forward", ()))

    assert len(created) == 4  # device 1's call created fresh events, device 0's pool was left alone
    assert len(timer._free_events[0]) == 2


def test_no_second_recording_starts_while_one_is_active():
    original_apply = nn.Module._apply
    structure, second = StructureRecorder(), StructureRecorder()
    timer = TimingRecorder(CallTree())

    with structure:
        with pytest.raises(RuntimeError, match="Another call recorder is already recording"):
            second.__enter__()
        with pytest.raises(RuntimeError, match="Another call recorder is already recording"):
            timer.__enter__()
        assert nn.Module._apply is not original_apply
    assert CallRecorder.active() is None
    assert nn.Module._apply is original_apply

    with timer:
        with pytest.raises(RuntimeError, match="Another call recorder is already recording"):
            structure.__enter__()
        with pytest.raises(RuntimeError, match="Another call recorder is already recording"):
            timer.__enter__()
    assert CallRecorder.active() is None
    assert nn.Module._apply is original_apply

    with pytest.raises(RuntimeError, match="boom"), timer:
        _boom()
    assert CallRecorder.active() is None


def _record_and_time(structure, model, call):
    """One structure-pass sample of ``call()``, then a timing pass over it, with the wrappers kept on."""
    with instrumented(structure, model):
        with structure:
            with structure.record_sample():
                call()
        tree = structure.result()
        with TimingRecorder(tree) as timer:
            for depths in timer.depths_per_sample():
                for depth in depths:
                    with timer.time_replay(depth):
                        call()
    return tree


def test_cpu_timing_is_recorded_in_the_timing_pass(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=False)
    model = TwoRoleToy()
    structure = StructureRecorder()
    x = torch.randn(2, 8)

    tree = _record_and_time(structure, model, lambda: model(x))

    root = tree.nodes[("root.forward",)]
    assert root.total_time_ms is not None and root.total_time_ms > 0
    assert tree.total_time_ms == root.total_time_ms


def test_accumulated_calls_double_the_measured_time(mocker):
    """Two structure passes on one recorder record twice the samples, so the second timed tree reads 2× the first.

    The clock advances by one second per read, so both passes measure the same per-call time.
    """
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=False)
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.time.perf_counter", side_effect=itertools.count())
    model = TwoRoleToy()
    structure = StructureRecorder()
    x = torch.randn(2, 8)

    first = _record_and_time(structure, model, lambda: model(x)).nodes[("root.forward",)].total_time_ms
    assert first is not None and first > 0

    second = _record_and_time(structure, model, lambda: model(x)).nodes[("root.forward",)].total_time_ms
    assert second == 2 * first


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA events require a GPU")
def test_the_timing_pass_writes_cuda_times_into_the_tree():
    model = TwoRoleToy().cuda()
    structure = StructureRecorder()

    x = torch.randn(2, 8, device="cuda")
    tree = _record_and_time(structure, model, lambda: model(x))
    first = tree.nodes[("root.forward",)].total_time_ms

    assert first is not None and first > 0
    assert tree.nodes[("root.forward",)].total_time_ms == first


def _tree_with_a_child() -> CallTree:
    """root -> child, one call each, no times yet."""
    root = CallNode(
        path=("root.forward",),
        label="root.forward",
        owner_class="Root",
        method="forward",
        calls_per_sample=[1],
    )
    child = CallNode(
        path=("root.forward", "root.child.forward"),
        label="root.child.forward",
        owner_class="Linear",
        method="forward",
        calls_per_sample=[1],
    )
    root.children.append(child.path)
    return CallTree(nodes={root.path: root, child.path: child})


def test_the_timing_pass_writes_times_into_the_tree(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=False)
    root_path, child_path = ("root.forward",), ("root.forward", "root.child.forward")
    tree = _tree_of([[root_path, child_path], [root_path, child_path]])
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        for _ in range(2):  # one replay per depth of each sample
            with timer.time_replay(1):
                timer.end_call(timer.begin_call("root.forward", model, "forward", ()))
            with timer.time_replay(2):
                root = timer.begin_call("root.forward", model, "forward", ())
                child = timer.begin_call("root.child.forward", model, "forward", ())
                timer.end_call(child)
                timer.end_call(root)

    assert all(node.total_time_ms is not None for node in tree.nodes.values())


def test_a_second_timing_pass_resets_nodes_it_did_not_time(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=False)
    root_path, child_path = ("root.forward",), ("root.forward", "root.child.forward")
    tree = _tree_of([[root_path, child_path], [root_path, child_path]])
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        for _ in range(2):
            with timer.time_replay(1):
                timer.end_call(timer.begin_call("root.forward", model, "forward", ()))
            with timer.time_replay(2):
                root = timer.begin_call("root.forward", model, "forward", ())
                child = timer.begin_call("root.child.forward", model, "forward", ())
                timer.end_call(child)
                timer.end_call(root)
    child_path = ("root.forward", "root.child.forward")
    assert tree.nodes[child_path].total_time_ms is not None

    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.logger.warning")
    with TimingRecorder(tree) as timer:  # this pass replays only the root
        for _ in range(2):
            with timer.time_replay(1):
                timer.end_call(timer.begin_call("root.forward", model, "forward", ()))

    assert tree.nodes[("root.forward",)].total_time_ms is not None
    assert tree.nodes[child_path].total_time_ms is None


def test_a_timing_pass_that_raises_leaves_all_times_none():
    tree = _tree_with_a_child()
    tree.nodes[("root.forward",)].total_time_ms = 5.0  # an earlier pass's time

    with pytest.raises(RuntimeError, match="boom"), TimingRecorder(tree):
        _boom()

    assert all(node.total_time_ms is None for node in tree.nodes.values())


def test_timing_a_copy_leaves_the_originals_times(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=False)
    tree = _tree_with_a_child()
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        with timer.time_replay(1):
            timer.end_call(timer.begin_call("root.forward", model, "forward", ()))
    original = tree.nodes[("root.forward",)].total_time_ms
    assert original is not None

    copied = tree.copy()
    with TimingRecorder(copied) as timer:
        with timer.time_replay(2):  # a replay that times depth 2, which the call below is not at
            timer.end_call(timer.begin_call("root.forward", model, "forward", ()))

    assert copied.nodes[("root.forward",)].total_time_ms is None
    assert tree.nodes[("root.forward",)].total_time_ms == original


def test_only_calls_at_the_timed_depth_are_timed(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=False)
    root_path, child_path = ("root.forward",), ("root.forward", "root.child.forward")
    tree = _tree_of([[root_path, child_path], [root_path]])
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        with timer.time_replay(2):  # this replay times depth 2: only the child is timed
            root = timer.begin_call("root.forward", model, "forward", ())
            child = timer.begin_call("root.child.forward", model, "forward", ())
            timer.end_call(child)
            timer.end_call(root)

    assert tree.nodes[("root.forward",)].total_time_ms is None
    assert tree.nodes[("root.forward", "root.child.forward")].total_time_ms is not None


def test_a_raised_replay_invalidates_the_pass(mocker):
    """A replay that raises invalidates the pass: nothing is written, not even times already taken."""
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=False)
    mocker.patch(
        "aitune.torch.call_tree.recording.timing_recorder.time.perf_counter",
        side_effect=_perf_reads([1.0, 1.0]),
    )
    a = ("a",)
    tree = _tree_of([[a]])
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        with pytest.raises(RuntimeError, match="boom"):
            with timer.time_replay(1):
                timer.end_call(timer.begin_call("a", model, "forward", ()))  # a completed, already-timed call
                _boom()
        with timer.time_replay(1):  # a successful replay cannot save the pass
            timer.end_call(timer.begin_call("a", model, "forward", ()))

    assert tree.nodes[a].total_time_ms is None


def test_a_raised_replay_publishes_no_cuda_times(mocker):
    """Same on the CUDA path: event pairs resolved after a raised replay are never written."""

    class FakeEvent:
        def __init__(self, *args, **kwargs):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, end):
            return 1.0

    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.Event", FakeEvent)
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder._record", side_effect=lambda event, stream: None)
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.current_device", return_value=0)
    mocker.patch.object(TimingRecorder, "_current_stream", lambda self, device: object())
    a = ("a",)
    tree = _tree_of([[a]])
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        timer._use_cuda = True
        with pytest.raises(RuntimeError, match="boom"):
            with timer.time_replay(1):
                timer.end_call(timer.begin_call("a", model, "forward", ()))
                _boom()
        with timer.time_replay(1):  # a successful replay cannot save the pass
            timer.end_call(timer.begin_call("a", model, "forward", ()))

    assert tree.nodes[a].total_time_ms is None


def test_the_timed_depth_resets_on_entry(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=False)
    tree = _tree_with_a_child()
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        with timer.time_replay(1):
            pass
    with TimingRecorder(tree) as timer:  # no time_replay: nothing is timed
        timer.end_call(timer.begin_call("root.forward", model, "forward", ()))

    assert tree.nodes[("root.forward",)].total_time_ms is None


def _warning_text(warning) -> str:
    warning.assert_called_once()
    return warning.call_args.args[0] % warning.call_args.args[1:]


def test_a_node_timed_for_fewer_calls_than_it_made_is_reported(mocker):
    """The timing pass replayed different calls than the structure pass saw: warn once, naming example paths."""
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=False)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing_recorder.logger.warning")
    a, ab = ("a",), ("a", "b")
    tree = _tree_of([[a, ab]])
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        with timer.time_replay(1):  # the depth-2 replay never happens, e.g. the replay did not call b
            timer.end_call(timer.begin_call("a", model, "forward", ()))

    text = _warning_text(warning)
    assert "timed a different number of calls than the structure pass recorded for 1 call-tree node" in text
    assert "a > b" in text
    assert "generate" in text


def test_a_node_inside_a_compiled_part_is_never_counted(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=False)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing_recorder.logger.warning")
    root, inner = ("a",), ("a", "b")
    tree = _tree_of([[root, inner]])
    tree.nodes[inner].inside_compiled_part = True  # the pass cannot time it; it never counts
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        with timer.time_replay(1):
            timer.end_call(timer.begin_call("a", model, "forward", ()))

    warning.assert_not_called()


def test_a_call_inside_a_compiled_part_is_never_timed(mocker):
    """A call reported with a compiled-part node's path gets no timer: its ``total_time_ms`` stays None."""
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=False)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing_recorder.logger.warning")
    a, ab, ac = ("a",), ("a", "b"), ("a", "c")
    tree = _tree_of([[a, ab, ac]])
    tree.nodes[ac].inside_compiled_part = True
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        with timer.time_replay(1):
            timer.end_call(timer.begin_call("a", model, "forward", ()))
        with timer.time_replay(2):
            a_call = timer.begin_call("a", model, "forward", ())
            timer.end_call(timer.begin_call("b", model, "forward", ()))
            timer.end_call(timer.begin_call("c", model, "forward", ()))
            timer.end_call(a_call)

    assert tree.nodes[ab].total_time_ms is not None
    assert tree.nodes[ac].total_time_ms is None
    warning.assert_not_called()


def test_a_complete_pass_does_not_warn(mocker):
    """Every call timed on the sample the structure pass recorded it on: nothing to report."""
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=False)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing_recorder.logger.warning")
    a, ab = ("a",), ("a", "b")
    tree = _tree_of([[a, ab]])
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        with timer.time_replay(1):
            timer.end_call(timer.begin_call("a", model, "forward", ()))
        with timer.time_replay(2):
            a_call = timer.begin_call("a", model, "forward", ())
            b_call = timer.begin_call("b", model, "forward", ())
            timer.end_call(b_call)
            timer.end_call(a_call)

    warning.assert_not_called()


def test_calls_below_the_timed_depth_are_skipped(mocker):
    """A call nested inside a timed call is skipped entirely — the recorder never records it."""
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=False)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing_recorder.logger.warning")
    root_path, child_path = ("root.forward",), ("root.forward", "root.child.forward")
    tree = _tree_of([[root_path, child_path]])
    model = nn.Linear(2, 2)

    def call(label):
        return timer.call(lambda: None, (), {}, label=label, obj=model, method="forward")

    with TimingRecorder(tree) as timer:
        with timer.time_replay(1):  # depth 1: the root runs and is timed; a nested child call is skipped
            timer.call(lambda: call("root.child.forward"), (), {}, label="root.forward", obj=model, method="forward")
            assert child_path not in timer._times  # the skip kept the child out of the recording entirely

        with timer.time_replay(2):  # the child's depth: this time the child is recorded and timed
            root = timer.begin_call("root.forward", model, "forward", ())
            child = timer.begin_call("root.child.forward", model, "forward", ())
            timer.end_call(child)
            timer.end_call(root)

    assert all(node.total_time_ms is not None for node in tree.nodes.values())
    warning.assert_not_called()


def test_a_replay_with_fewer_calls_keeps_their_times(mocker):
    """The replay timed fewer calls than the structure pass recorded: the measured times are kept."""
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=False)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing_recorder.logger.warning")
    mocker.patch(
        "aitune.torch.call_tree.recording.timing_recorder.time.perf_counter",
        side_effect=_perf_reads([2.0, 6.0]),
    )
    a = ("a",)
    tree = _tree_of([[a, a], [a]])  # a's calls_per_sample: [2, 1]; the replays run one call each
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        for _ in range(2):
            with timer.time_replay(1):
                timer.end_call(timer.begin_call("a", model, "forward", ()))

    node = tree.nodes[a]
    assert node.total_time_ms == pytest.approx(8.0)
    text = _warning_text(warning)
    assert "total_time_ms only counts the calls that were timed" in text


def test_a_stray_exit_does_not_release_the_active_recorder():
    other = TimingRecorder(CallTree())
    with TimingRecorder(CallTree()) as active:
        other.__exit__(None, None, None)
        assert CallRecorder.active() is active
    assert CallRecorder.active() is None


def test_a_transparent_container_gives_its_children_the_same_path_in_both_passes(mocker):
    """``nn.Sequential`` records no node; its children get the same path and time in both passes."""
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=False)
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    x = torch.randn(2, 8)
    structure = StructureRecorder()

    with instrumented(structure, model):
        with structure:
            with structure.record_sample():
                model(x)
        tree = structure.result()
        warning = mocker.patch("aitune.torch.call_tree.recording.timing_recorder.logger.warning")
        with TimingRecorder(tree) as timer:
            for depths in timer.depths_per_sample():
                for depth in depths:
                    with timer.time_replay(depth):
                        model(x)

    warning.assert_not_called()
    assert ("root.forward",) not in tree.nodes
    assert tree.nodes[("root.0.forward",)].total_time_ms is not None


def test_max_depth_leaves_deeper_nodes_untimed_without_a_warning(mocker):
    """Deeper nodes are still recorded by the structure pass; the timing pass just never plans their depths."""
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=False)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing_recorder.logger.warning")
    a, ab, abc = ("a",), ("a", "b"), ("a", "b", "c")
    tree = _tree_of([[a, ab, abc]] * 2)
    model = nn.Linear(2, 2)

    with TimingRecorder(tree, max_depth=1) as timer:
        for _ in range(2):
            with timer.time_replay(1):  # every replay times depth 1 of its sample
                timer.end_call(timer.begin_call("a", model, "forward", ()))

    assert tree.nodes[a].total_time_ms is not None
    assert tree.nodes[abc].total_time_ms is None
    warning.assert_not_called()


def test_a_node_deeper_than_max_depth_is_never_counted(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=False)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing_recorder.logger.warning")
    a, ab, abc = ("a",), ("a", "b"), ("a", "b", "c")
    tree = _tree_of([[a, ab, abc]])
    model = nn.Linear(2, 2)

    with TimingRecorder(tree, max_depth=1) as timer:
        with timer.time_replay(1):  # only depth 1 is timed; the deeper nodes are off limits, not untimed
            timer.end_call(timer.begin_call("a", model, "forward", ()))

    warning.assert_not_called()


def test_a_replay_with_more_calls_keeps_their_times(mocker):
    """The replay timed more calls than the structure pass recorded: the measured times are kept."""
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=False)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing_recorder.logger.warning")
    mocker.patch(
        "aitune.torch.call_tree.recording.timing_recorder.time.perf_counter",
        side_effect=_perf_reads([1.0, 1.0, 1.0]),
    )
    a = ("a",)
    tree = _tree_of([[a, a]])
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        with timer.time_replay(1):
            for _ in range(3):  # the replay made three of a's two recorded calls
                timer.end_call(timer.begin_call("a", model, "forward", ()))

    assert tree.nodes[a].total_time_ms == pytest.approx(3.0)
    assert "total_time_ms only counts the calls that were timed" in _warning_text(warning)


def test_untimed_nodes_have_no_times(mocker):
    """Inside a compiled part or deeper than ``max_depth``, every time field stays None."""
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=False)
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.logger.warning")
    a, ab, abc = ("a",), ("a", "b"), ("a", "b", "c")
    tree = _tree_of([[a, ab, abc]])
    tree.nodes[ab].inside_compiled_part = True
    model = nn.Linear(2, 2)

    with TimingRecorder(tree, max_depth=2) as timer:
        with timer.time_replay(1):
            timer.end_call(timer.begin_call("a", model, "forward", ()))
        with timer.time_replay(2):
            a_call = timer.begin_call("a", model, "forward", ())
            timer.end_call(timer.begin_call("b", model, "forward", ()))
            timer.end_call(a_call)

    for node in (tree.nodes[ab], tree.nodes[abc]):
        assert node.total_time_ms is None


def test_a_second_pass_replaces_the_firsts_times(mocker):
    """Entering a pass resets ``total_time_ms``, so a second pass replaces the first's times."""

    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.torch.cuda.is_available", return_value=False)
    mocker.patch("aitune.torch.call_tree.recording.timing_recorder.logger.warning")
    mocker.patch(
        "aitune.torch.call_tree.recording.timing_recorder.time.perf_counter",
        side_effect=_perf_reads([1.0, 2.0]),
    )
    a = ("a",)
    tree = _tree_of([[a]])
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        with timer.time_replay(1):
            timer.end_call(timer.begin_call("a", model, "forward", ()))
    node = tree.nodes[a]
    assert node.total_time_ms is not None

    with TimingRecorder(tree) as timer:
        with timer.time_replay(2):  # a replay timing depth 2, which the reported call is not at
            timer.end_call(timer.begin_call("a", model, "forward", ()))

    assert node.total_time_ms is None
