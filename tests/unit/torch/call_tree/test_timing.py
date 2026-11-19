# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import threading
from collections import Counter

import pytest
import torch
from torch import nn

from aitune.torch.call_tree.call_tree import CallTree
from aitune.torch.call_tree.nodes import CallNode, TimedIteration
from aitune.torch.call_tree.recording import CallRecorder, StructureRecorder, TimingRecorder, depth_schedule
from tests.toy_models.call_tree_models import TwoRoleToy
from tests.unit.torch.call_tree.helpers import instrumented


def _tree_of(sequences: list[list[tuple[str, ...]]]) -> CallTree:
    """A call tree whose nodes are exactly the paths ``sequences`` lists, one call per occurrence."""
    nodes = {}
    for sequence in sequences:
        for path in sequence:
            if path not in nodes:
                nodes[path] = CallNode(path=path, label=path[-1], owner_class="M", method="forward", calls=0)
            nodes[path].calls += 1
    return CallTree(nodes=nodes, iterations=len(sequences), sequences=sequences)


def _boom() -> None:
    raise RuntimeError("boom")


def test_every_iteration_times_the_shallowest_untimed_depth():
    paths: list[tuple[str, ...]] = [("a",), ("a", "b"), ("a", "b", "c")]
    tree = _tree_of([paths for _ in range(5)])

    assert depth_schedule(tree) == [1, 2, 3, 1, 2]


def test_a_path_called_once_in_each_of_more_iterations_has_more_chances():
    """Chances count later iterations containing the path, not calls: a path called three times in one later
    sample has fewer chances left than a path called once in each of two."""
    a, b, c = ("a",), ("a", "b"), ("a", "b", "c")
    tree = _tree_of([[a, b, c], [a, b, c, c, c], [a, b]])

    assert depth_schedule(tree) == [3, 1, 2]


def test_fewer_iterations_leave_the_deepest_paths_unscheduled():
    paths: list[tuple[str, ...]] = [("a",), ("a", "b"), ("a", "b", "c")]
    tree = _tree_of([paths for _ in range(2)])

    assert depth_schedule(tree) == [1, 2]


def test_a_call_only_some_samples_make_is_scheduled():
    root, child = ("a",), ("a", "b")
    tree = _tree_of([[root, child], [root], [root, child], [root]])

    schedule = depth_schedule(tree)

    # the child has only two chances, so its depth is timed before the root's second call
    assert schedule == [2, 1, 1, 1]
    assert child not in tree.sequences[1] and child not in tree.sequences[3]


def test_the_last_chance_of_an_untimed_call_wins_over_a_shallower_depth():
    root, child = ("a",), ("a", "b")
    tree = _tree_of([[root, child], [root], [root]])

    # the child is never called again: depth 2 wins over the shallower depth 1, which has chances left
    assert depth_schedule(tree) == [2, 1, 1]


def test_the_depth_with_the_fewer_remaining_chances_wins():
    a, ab, ac, acd = ("a",), ("a", "b"), ("a", "c"), ("a", "c", "d")
    tree = _tree_of([[a, ab, ac, acd], [a, ab], [a, ab, ac, acd], [a, ab]])

    # iteration 0: depth 2's untimed nodes have 1 chance left (b) and depth 3's has 1 too — shallowest wins;
    # plain shallowest-first would have taken depth 1 (3 chances left)
    assert depth_schedule(tree) == [2, 1, 3, 1]


def test_an_iteration_without_calls_gets_no_depth():
    tree = _tree_of([[("a",)], []])

    assert depth_schedule(tree) == [1, None]


def test_calls_inside_a_timed_call_are_skipped(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    tree = _tree_with_a_child()
    model = nn.Linear(2, 2)
    ran = []
    stack_during_root = []

    with TimingRecorder(tree) as timer:
        timer.begin_iteration()  # the schedule times depth 1: the root

        def child_forward():
            ran.append(True)

        def root_forward():
            stack_during_root.append(len(timer._stack))
            # a call inside a timed call is deeper than the timed depth: it runs but is not recorded
            timer.call(child_forward, (), {}, label="root.child.forward", obj=model, method="forward")

        timer.call(root_forward, (), {}, label="root.forward", obj=model, method="forward")

    child_path = ("root.forward", "root.child.forward")
    assert ran and stack_during_root == [1]
    assert child_path not in timer._calls
    assert timer._stack == []
    assert tree.nodes[("root.forward",)].time_ms is not None
    assert tree.nodes[child_path].time_ms is None


def test_the_structure_recorder_does_not_skip_calls():
    assert not StructureRecorder()._skips_calls()


def test_an_iteration_beyond_the_recorded_ones_times_nothing(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing.logger.warning")
    tree = _tree_with_a_child()
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        timer.begin_iteration()  # the only recorded iteration: depth 1
        assert timer._depth == 1
        timer.begin_iteration()  # the structure pass recorded no second iteration: time nothing
        assert timer._depth is None
        root = timer.begin_call("root.forward", model, "forward", ())
        child = timer.begin_call("root.child.forward", model, "forward", ())
        timer.end_call(child)
        timer.end_call(root)
        assert timer._expected_calls() == {("root.forward",): Counter({0: 1})}

    assert all(node.time_ms is None for node in tree.nodes.values())
    assert "did not run in the timing pass" not in _warning_text(warning)


def _perf_reads(durations_ms: list[float]) -> list[float]:
    """``perf_counter`` reads (seconds) that make consecutive calls take ``durations_ms`` each: 0, then the end."""
    return [t for d in durations_ms for t in (0.0, d / 1000)]


def test_the_estimate_is_the_median_of_the_per_iteration_mean_times(mocker):
    """A systematically slow first call per iteration must reach the estimate: mean per iteration, median across."""
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    mocker.patch(
        "aitune.torch.call_tree.recording.timing.time.perf_counter",
        side_effect=_perf_reads([4.0, 1.0, 1.0] * 3),  # calls of 4, 1 and 1 ms in each of 3 iterations
    )
    a = ("a",)
    tree = _tree_of([[a, a, a]] * 3)
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        for _ in range(3):
            timer.begin_iteration()
            for _ in range(3):
                timer.end_call(timer.begin_call("a", model, "forward", ()))

    node = tree.nodes[a]
    assert node.timings == [TimedIteration(i, 3, 6.0) for i in range(3)]
    # mean per iteration 2 ms; a flat per-call median over all nine calls would give 1 ms
    assert node.time_ms == pytest.approx(2.0 * node.calls)


def test_the_estimate_ignores_a_slow_iteration(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    mocker.patch(
        "aitune.torch.call_tree.recording.timing.time.perf_counter",
        side_effect=_perf_reads([1.0, 3.0, 1.0, 3.0, 18.0, 22.0]),  # per-iteration means 2 ms, 2 ms, 20 ms
    )
    a = ("a",)
    tree = _tree_of([[a], [a], [a]])
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        for _ in range(3):
            timer.begin_iteration()
            for _ in range(2):
                timer.end_call(timer.begin_call("a", model, "forward", ()))

    node = tree.nodes[a]
    assert node.time_ms == pytest.approx(2.0 * node.calls)  # the median drops the 20 ms outlier iteration


def test_time_ms_is_none_without_timings():
    node = CallNode(path=("a",), label="a", owner_class="M", method="forward", calls=2)

    assert node.time_ms is None


def test_time_ms_is_the_mean_per_call_times_calls_with_few_timed_iterations():
    node = CallNode(path=("a",), label="a", owner_class="M", method="forward", calls=4)
    node.timings = [TimedIteration(0, 1, 4.0)]
    assert node.time_ms == 16.0  # 4 ms per call times 4 calls

    node.timings = [TimedIteration(0, 1, 4.0), TimedIteration(1, 1, 6.0)]
    assert node.time_ms == 20.0  # mean of the per-call times (4 and 6 ms) times 4 calls


def test_time_ms_is_the_median_per_call_times_calls_with_three_timed_iterations():
    node = CallNode(path=("a",), label="a", owner_class="M", method="forward", calls=3)
    node.timings = [TimedIteration(0, 1, 4.0), TimedIteration(1, 1, 20.0), TimedIteration(2, 1, 6.0)]

    assert node.time_ms == 18.0  # median per call 6 ms times 3 calls


def test_the_pass_writes_one_timed_iteration_per_iteration_that_timed_the_node(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    # timed calls: the root in iterations 0 and 2 (2 ms, 6 ms), the child twice in iteration 1 (1 ms, 3 ms)
    mocker.patch(
        "aitune.torch.call_tree.recording.timing.time.perf_counter",
        side_effect=_perf_reads([2.0, 1.0, 3.0, 6.0]),
    )
    root_path, child_path = ("root.forward",), ("root.forward", "root.child.forward")
    tree = _tree_of([[root_path, child_path], [root_path, child_path, child_path], [root_path, child_path]])
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:  # the schedule times depths 1, 2, 1
        for _ in range(3):
            timer.begin_iteration()
            root = timer.begin_call("root.forward", model, "forward", ())
            if timer._depth is not None and timer._depth >= 2:  # below a timed depth the child is skipped
                for _ in range(tree.sequences[timer._iteration - 1].count(child_path)):
                    timer.end_call(timer.begin_call("root.child.forward", model, "forward", ()))
            timer.end_call(root)

    assert tree.nodes[root_path].timings == [TimedIteration(0, 1, 2.0), TimedIteration(2, 1, 6.0)]
    assert tree.nodes[child_path].timings == [TimedIteration(1, 2, 4.0)]


def _recorded_tree(model, x):
    """The tree of one structure-pass iteration of ``model(x)``."""
    recorder = StructureRecorder()
    with instrumented(recorder, model), recorder:
        recorder.begin_iteration()
        model(x)
        recorder.end_iteration()
    return recorder.result()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA events require a GPU")
def test_flush_resolves_pending_events_into_samples():
    model = nn.Linear(2, 2).cuda()
    x = torch.randn(2, 2, device="cuda")
    tree = _recorded_tree(model, x)

    with TimingRecorder(tree) as timer:
        timer.begin_iteration()
        call = timer.begin_call("root.forward", model, "forward", ())
        model(x)
        timer.end_call(call)

        assert timer._events
        timer.flush()
        assert not timer._events
        assert timer._free_events  # resolved events are reused

    node = tree.nodes[("root.forward",)]
    assert node.time_ms is not None and node.time_ms > 0


def test_a_model_on_the_cpu_is_timed_with_perf_counter(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=True)
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_initialized", return_value=True)
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.memory_reserved", return_value=0)
    event = mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.Event")
    model = nn.Linear(2, 2)
    x = torch.randn(2, 2)
    tree = _recorded_tree(model, x)

    with TimingRecorder(tree) as timer:
        assert not timer._use_cuda
        timer.begin_iteration()
        timer.end_call(timer.begin_call("root.forward", model, "forward", ()))

    event.assert_not_called()
    assert tree.nodes[("root.forward",)].time_ms is not None


def test_a_model_in_a_process_with_cuda_memory_is_timed_with_cuda_events(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=True)
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_initialized", return_value=True)
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.memory_reserved", return_value=1)
    model = nn.Linear(2, 2)
    x = torch.randn(2, 2)
    tree = _recorded_tree(model, x)

    with TimingRecorder(tree) as timer:
        assert timer._use_cuda


def test_cuda_memory_reserved_but_nothing_allocated_still_uses_cuda_events(mocker):
    """Freed CUDA tensors keep the allocator's memory reserved: earlier CUDA work still means CUDA events."""
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=True)
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_initialized", return_value=True)
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.memory_allocated", return_value=0)
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.memory_reserved", return_value=1)
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

    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.Event", FakeEvent)
    mocker.patch(
        "aitune.torch.call_tree.recording.timing._record",
        side_effect=lambda event, stream: setattr(event, "stream", stream),
    )
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.current_device", return_value=0)
    current = ["stream0"]
    mocker.patch.object(TimingRecorder, "_current_stream", lambda self, device: current[0])
    model = nn.Linear(2, 2)
    x = torch.randn(2, 2)
    tree = _recorded_tree(model, x)

    with TimingRecorder(tree) as timer:
        timer._use_cuda = True
        timer.begin_iteration()
        call = timer.begin_call("root.forward", model, "forward", ())
        current[0] = "stream1"  # the call switched the current stream
        timer.end_call(call)
        timer.flush()

    start, end = events
    assert start.stream == "stream0"
    assert end.stream == "stream0"
    assert tree.nodes[("root.forward",)].time_ms == 2.0


def test_flush_synchronizes_every_end_event(mocker):
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

    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.Event", FakeEvent)
    mocker.patch("aitune.torch.call_tree.recording.timing._record", side_effect=lambda event, stream: None)
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.current_device", return_value=0)
    mocker.patch.object(TimingRecorder, "_current_stream", lambda self, device: object())
    cuda_synchronize = mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.synchronize")
    model = nn.Linear(2, 2)
    x = torch.randn(2, 2)
    tree = _recorded_tree(model, x)

    with TimingRecorder(tree) as timer:
        timer._use_cuda = True
        timer.begin_iteration()
        for _ in range(2):
            timer.end_call(timer.begin_call("root.forward", model, "forward", ()))
        timer.flush()

    node = tree.nodes[("root.forward",)]
    assert node.time_ms == 1.0 * node.calls  # mean of the 1.0 samples, per call times the call count
    cuda_synchronize.assert_not_called()


def test_unflushed_cuda_samples_keep_their_iteration(mocker):
    elapsed = iter([1.0, 1.0, 100.0])

    class FakeEvent:
        def __init__(self, *args, **kwargs):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, end):
            return next(elapsed)

    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.Event", FakeEvent)
    mocker.patch("aitune.torch.call_tree.recording.timing._record", side_effect=lambda event, stream: None)
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.current_device", return_value=0)
    mocker.patch.object(TimingRecorder, "_current_stream", lambda self, device: object())
    a = ("a",)
    tree = _tree_of([[a], [a], [a]])
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        timer._use_cuda = True
        for _ in range(3):
            timer.begin_iteration()
            timer.end_call(timer.begin_call("a", model, "forward", ()))
        # no flush between the iterations: each event pair still lands under its own iteration

    node = tree.nodes[a]
    assert node.timings == [TimedIteration(0, 1, 1.0), TimedIteration(1, 1, 1.0), TimedIteration(2, 1, 100.0)]
    assert node.time_ms == 1.0 * node.calls  # the median per call drops the 100 ms iteration


def test_flushed_events_are_reused_only_on_their_device(mocker):
    """The event pool is per device: a pair resolved on device 0 must not serve a call on device 1."""
    created = []

    class FakeEvent:
        def __init__(self, *args, **kwargs):
            created.append(self)

        def synchronize(self):
            pass

        def elapsed_time(self, end):
            return 1.0

    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.Event", FakeEvent)
    mocker.patch("aitune.torch.call_tree.recording.timing._record", side_effect=lambda event, stream: None)
    device = [0]
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.current_device", side_effect=lambda: device[0])
    mocker.patch.object(TimingRecorder, "_current_stream", lambda self, dev: f"stream{dev}")
    a = ("a",)
    tree = _tree_of([[a], [a]])
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        timer._use_cuda = True
        timer.begin_iteration()
        timer.end_call(timer.begin_call("a", model, "forward", ()))
        timer.flush()
        assert timer._free_events[0] == created[:2]  # the flushed pair went back to device 0's pool

        device[0] = 1
        timer.begin_iteration()
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
    """One structure-pass iteration of ``call()``, then a timing pass over it, with the wrappers kept on."""
    with instrumented(structure, model):
        with structure:
            structure.begin_iteration()
            call()
            structure.end_iteration()
        tree = structure.result()
        with TimingRecorder(tree) as timer:
            for _ in tree.sequences:
                timer.begin_iteration()
                call()
                timer.flush()
    return tree


def test_cpu_timing_is_recorded_in_the_timing_pass(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    model = TwoRoleToy()
    structure = StructureRecorder()
    x = torch.randn(2, 8)

    tree = _record_and_time(structure, model, lambda: model(x))

    root = tree.nodes[("root.forward",)]
    assert root.time_ms is not None and root.time_ms > 0
    assert tree.total_ms == root.time_ms


def test_accumulated_calls_double_the_estimated_time(mocker):
    """Two structure passes on one recorder double ``calls``, so the second timed tree reads 2× the first.

    The clock advances by one second per read, so both passes measure the same per-call time.
    """
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    mocker.patch("aitune.torch.call_tree.recording.timing.time.perf_counter", side_effect=itertools.count())
    model = TwoRoleToy()
    structure = StructureRecorder()
    x = torch.randn(2, 8)

    first = _record_and_time(structure, model, lambda: model(x)).nodes[("root.forward",)].time_ms
    assert first is not None and first > 0

    second = _record_and_time(structure, model, lambda: model(x)).nodes[("root.forward",)].time_ms
    assert second == 2 * first


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA events require a GPU")
def test_the_timing_pass_writes_cuda_times_into_the_tree():
    model = TwoRoleToy().cuda()
    structure = StructureRecorder()

    x = torch.randn(2, 8, device="cuda")
    tree = _record_and_time(structure, model, lambda: model(x))
    first = tree.nodes[("root.forward",)].time_ms

    assert first is not None and first > 0
    assert tree.nodes[("root.forward",)].time_ms == first


def _tree_with_a_child() -> CallTree:
    """root -> child, one call each, no times yet."""
    root = CallNode(path=("root.forward",), label="root.forward", owner_class="Root", method="forward", calls=1)
    child = CallNode(
        path=("root.forward", "root.child.forward"),
        label="root.child.forward",
        owner_class="Linear",
        method="forward",
        calls=1,
    )
    root.children.append(child.path)
    return CallTree(nodes={root.path: root, child.path: child}, iterations=1, sequences=[[root.path, child.path]])


def test_the_timing_pass_writes_times_into_the_tree(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    root_path, child_path = ("root.forward",), ("root.forward", "root.child.forward")
    tree = _tree_of([[root_path, child_path], [root_path, child_path]])
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:  # the schedule times depth 1, then depth 2
        timer.begin_iteration()
        timer.end_call(timer.begin_call("root.forward", model, "forward", ()))
        timer.begin_iteration()
        root = timer.begin_call("root.forward", model, "forward", ())
        child = timer.begin_call("root.child.forward", model, "forward", ())
        timer.end_call(child)
        timer.end_call(root)

    assert all(node.time_ms is not None for node in tree.nodes.values())


def test_a_second_timing_pass_resets_nodes_it_did_not_time(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    root_path, child_path = ("root.forward",), ("root.forward", "root.child.forward")
    tree = _tree_of([[root_path, child_path], [root_path, child_path]])
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:  # the schedule times depth 1, then depth 2
        timer.begin_iteration()
        timer.end_call(timer.begin_call("root.forward", model, "forward", ()))
        timer.begin_iteration()
        root = timer.begin_call("root.forward", model, "forward", ())
        child = timer.begin_call("root.child.forward", model, "forward", ())
        timer.end_call(child)
        timer.end_call(root)
    child_path = ("root.forward", "root.child.forward")
    assert tree.nodes[child_path].time_ms is not None

    with TimingRecorder(tree) as timer:  # this pass calls only the root
        timer.begin_iteration()
        timer.end_call(timer.begin_call("root.forward", model, "forward", ()))

    assert tree.nodes[("root.forward",)].time_ms is not None
    assert tree.nodes[child_path].time_ms is None


def test_a_timing_pass_that_raises_leaves_all_times_none():
    tree = _tree_with_a_child()
    tree.nodes[("root.forward",)].timings = [TimedIteration(0, 1, 5.0)]  # an earlier pass's time

    with pytest.raises(RuntimeError, match="boom"), TimingRecorder(tree):
        _boom()

    assert all(node.time_ms is None for node in tree.nodes.values())


def test_timing_a_copy_leaves_the_originals_times(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    tree = _tree_with_a_child()
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        timer.begin_iteration()
        timer.end_call(timer.begin_call("root.forward", model, "forward", ()))
    original = tree.nodes[("root.forward",)].time_ms
    assert original is not None

    copied = tree.copy()
    with TimingRecorder(copied) as timer:
        timer.begin_iteration()
        timer.begin_iteration()  # beyond the recorded iterations: times nothing
        timer.end_call(timer.begin_call("root.forward", model, "forward", ()))

    assert copied.nodes[("root.forward",)].time_ms is None
    assert tree.nodes[("root.forward",)].time_ms == original


def test_only_calls_at_the_selected_depth_are_timed(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    root_path, child_path = ("root.forward",), ("root.forward", "root.child.forward")
    # the child's only later chance is this iteration, so the schedule times depth 2 first
    tree = _tree_of([[root_path, child_path], [root_path]])
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        timer.begin_iteration()  # depth 2: only the child is timed
        root = timer.begin_call("root.forward", model, "forward", ())
        child = timer.begin_call("root.child.forward", model, "forward", ())
        timer.end_call(child)
        timer.end_call(root)

    assert tree.nodes[("root.forward",)].time_ms is None
    assert tree.nodes[("root.forward", "root.child.forward")].time_ms is not None


def test_the_iteration_count_resets_on_entry(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    root_path, child_path = ("root.forward",), ("root.forward", "root.child.forward")
    tree = _tree_of([[root_path, child_path], [root_path, child_path]])

    with TimingRecorder(tree) as timer:
        timer.begin_iteration()
        timer.begin_iteration()
        assert timer._depth == 2
    with TimingRecorder(tree) as timer:
        timer.begin_iteration()
        assert timer._depth == 1  # the schedule restarts from its first iteration


def test_the_timed_depth_resets_on_entry(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    tree = _tree_with_a_child()
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        timer.begin_iteration()
    with TimingRecorder(tree) as timer:  # no begin_iteration: nothing is timed
        timer.end_call(timer.begin_call("root.forward", model, "forward", ()))

    assert tree.nodes[("root.forward",)].time_ms is None


def _warning_text(warning) -> str:
    warning.assert_called_once()
    return warning.call_args.args[1]


def test_a_node_that_did_not_run_is_reported(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing.logger.warning")
    tree = _tree_of([[("a",), ("a", "b")], [("a",)]])
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:  # the child never runs although a timed depth covers it
        for _ in range(2):  # the schedule times depth 2, then depth 1
            timer.begin_iteration()
            timer.end_call(timer.begin_call("a", model, "forward", ()))

    text = _warning_text(warning)
    assert "1 of the call tree's 2 nodes have no time" in text
    assert "did not run in the timing pass" in text
    assert "a > b" in text


def test_a_depth_the_pass_never_timed_is_reported_with_the_needed_iterations(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing.logger.warning")
    tree = _tree_of([[("a",), ("a", "b"), ("a", "b", "c")]] * 2)
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        for _ in range(2):  # the schedule times depths 1 and 2; depth 3 is never timed
            timer.begin_iteration()
            a = timer.begin_call("a", model, "forward", ())
            if timer._depth is not None and timer._depth >= 2:
                b = timer.begin_call("b", model, "forward", ())
                timer.end_call(b)
            timer.end_call(a)

    text = _warning_text(warning)
    assert "1 of the call tree's 3 nodes have no time" in text
    assert "number_of_iterations must be at least 3" in text
    assert "a > b > c" in text


def test_calls_below_the_timed_depth_are_skipped_and_the_node_is_reported(mocker):
    """A call nested inside a timed call is skipped entirely — the recorder never counts it — so the node the
    structure pass recorded is reported as not run when no timed depth ever reaches it."""
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing.logger.warning")
    root_path, child_path = ("root.forward",), ("root.forward", "root.child.forward")
    tree = _tree_of([[root_path, child_path]] * 3)  # the schedule times depths 1, 2, 1
    model = nn.Linear(2, 2)

    def call(label):
        return timer.call(lambda: None, (), {}, label=label, obj=model, method="forward")

    with TimingRecorder(tree) as timer:
        timer.begin_iteration()  # depth 1: the root runs and is timed; a nested child call is skipped
        timer.call(lambda: call("root.child.forward"), (), {}, label="root.forward", obj=model, method="forward")
        timer.begin_iteration()  # depth 2: the child's depth is timed, but the child does not run
        call("root.forward")
        timer.begin_iteration()  # depth 1 again
        call("root.forward")

    assert child_path not in timer._calls  # the skip kept the child out of the counts entirely
    assert tree.nodes[child_path].time_ms is None
    text = _warning_text(warning)
    assert "did not run in the timing pass" in text
    assert "child.forward" in text


def test_a_node_without_a_time_is_listed_when_its_depth_was_timed_elsewhere(mocker):
    """``a > x``'s depth is timed, but only in an iteration ``a > x`` never runs: it is reported, not skipped."""
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing.logger.warning")
    a, ax, b, by = ("a",), ("a", "x"), ("b",), ("b", "y")
    tree = _tree_of([[a, ax], [b, by], [b, by]])
    assert depth_schedule(tree) == [1, 1, 2]  # depth 2 is timed only where ``a > x`` never runs
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        timer.begin_iteration()  # depth 1: a runs; a > x inside it is skipped
        timer.end_call(timer.begin_call("a", model, "forward", ()))
        timer.begin_iteration()  # depth 1: b runs
        timer.end_call(timer.begin_call("b", model, "forward", ()))
        timer.begin_iteration()  # depth 2: b and b > y run; a > x does not
        frame = timer.begin_call("b", model, "forward", ())
        timer.end_call(timer.begin_call("y", model, "forward", ()))
        timer.end_call(frame)

    node = tree.nodes[ax]
    assert node.time_ms is None
    text = _warning_text(warning)
    assert "a > x" in text


def test_a_different_call_count_is_reported_but_the_node_is_timed(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing.logger.warning")
    root_path, child_path = ("root.forward",), ("root.forward", "root.child.forward")
    tree = _tree_of([[root_path, child_path], [root_path, child_path]])
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        timer.begin_iteration()  # depth 1: only the root is counted
        timer.end_call(timer.begin_call("root.forward", model, "forward", ()))
        timer.begin_iteration()  # depth 2 covers the child; it runs one time too many
        root = timer.begin_call("root.forward", model, "forward", ())
        for _ in range(2):
            timer.end_call(timer.begin_call("root.child.forward", model, "forward", ()))
        timer.end_call(root)

    assert tree.nodes[child_path].time_ms is not None
    text = _warning_text(warning)
    assert "ran a different number of times than in the structure pass" in text
    assert "child.forward" in text


def test_a_different_call_count_in_one_iteration_is_reported(mocker):
    """The same total shifted between iterations — 1 then 3 recorded, 3 then 1 timed — still differs."""
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing.logger.warning")
    a = ("root.forward",)
    tree = _tree_of([[a], [a, a, a]])
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:  # the schedule times depth 1 in both iterations; every call is counted
        timer.begin_iteration()
        for _ in range(3):  # runs 3 times where the structure pass made 1 call
            timer.end_call(timer.begin_call("root.forward", model, "forward", ()))
        timer.begin_iteration()
        timer.end_call(timer.begin_call("root.forward", model, "forward", ()))  # once where it made 3

    text = _warning_text(warning)
    assert "ran a different number of times than in the structure pass" in text
    assert text.count("forward") == 1  # one entry for the path (rendered "forward"), not one per iteration


def test_a_call_without_a_recorded_node_is_reported(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing.logger.warning")
    tree = _tree_with_a_child()
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        timer.begin_iteration()
        timer.end_call(timer.begin_call("root.forward", model, "forward", ()))
        timer.end_call(timer.begin_call("root.unknown.forward", model, "forward", ()))

    text = _warning_text(warning)
    assert "calls in the timing pass have no recorded node" in text
    assert "unknown.forward" in text


def _tree_with_a_compiled_module() -> tuple[CallTree, nn.Module]:
    """root -> child where the root's module is compiled (``nn.Module.compile()``) and the child is inside it."""
    compiled = nn.Sequential(nn.Linear(2, 2))
    compiled._compiled_call_impl = lambda *args, **kwargs: None  # what nn.Module.compile() sets
    root = CallNode(
        path=("root.forward",),
        label="root.forward",
        owner_class="Seq",
        method="forward",
        module=compiled,
        calls=1,
    )
    child = CallNode(
        path=("root.forward", "root.child.forward"),
        label="root.child.forward",
        owner_class="Linear",
        method="forward",
        module=compiled[0],
        calls=1,
    )
    root.children.append(child.path)
    return (
        CallTree(
            nodes={root.path: root, child.path: child},
            iterations=1,
            sequences=[[root.path, child.path]],
            modules=[("root", compiled)],
        ),
        compiled,
    )


def test_calls_inside_a_compiled_part_are_not_reported(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing.logger.warning")
    tree, compiled = _tree_with_a_compiled_module()

    with TimingRecorder(tree) as timer:  # the compiled module runs as one call; the call inside never does
        timer.begin_iteration()
        timer.end_call(timer.begin_call("root.forward", compiled, "forward", ()))

    assert tree.nodes[("root.forward",)].time_ms is not None
    warning.assert_not_called()


def test_a_compiled_modules_other_methods_do_not_hide_calls_below_them(mocker):
    """``compile()`` compiles the call (``forward``) only: another method of the compiled module runs eagerly,
    so an untimed call below it is reported like any other."""
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing.logger.warning")
    tree, compiled = _tree_with_a_compiled_module()
    tree.nodes[("root.forward",)].method = "encode"  # the same compiled module, but a user-defined method

    with TimingRecorder(tree) as timer:  # nothing runs: the call below the eager method is reported too
        timer.begin_iteration()

    text = _warning_text(warning)
    assert "child" in text


def test_an_optimized_module_is_the_exclusion_root(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing.logger.warning")
    inner = nn.Sequential(nn.Linear(2, 2))
    optimized = torch.compile(inner, backend="eager")
    assert isinstance(optimized, nn.Module)
    root = CallNode(
        path=("root.forward",), label="root.forward", owner_class="Seq", method="forward", module=optimized, calls=1
    )
    child = CallNode(
        path=("root.forward", "root.child.forward"),
        label="root.child.forward",
        owner_class="Linear",
        method="forward",
        module=inner[0],
        calls=1,
    )
    root.children.append(child.path)
    tree = CallTree(
        nodes={root.path: root, child.path: child},
        iterations=1,
        sequences=[[root.path, child.path]],
        modules=[("b", optimized)],
    )

    with TimingRecorder(tree) as timer:  # nothing runs: the ``OptimizedModule`` node is reported, the inside not
        timer.begin_iteration()

    text = _warning_text(warning)
    assert "did not run" in text
    assert "child" not in text


def test_an_eager_call_of_a_module_inside_a_compiled_part_is_reported(mocker):
    """A module that is called eagerly *and* inside a compiled part: only the path below the compiled call is
    inside, so the eager call — untimed here — is reported, not silenced."""
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing.logger.warning")
    shared = nn.Linear(2, 2)
    compiled = nn.Sequential(shared)  # ``shared`` is a strict submodule of the compiled module
    compiled._compiled_call_impl = lambda *args, **kwargs: None
    paths = {
        "root": ("root.forward",),
        "eager": ("root.forward", "root.eager.forward"),
        "comp": ("root.forward", "root.comp.forward"),
        "inner": ("root.forward", "root.comp.forward", "root.comp.inner.forward"),
    }
    root = CallNode(path=paths["root"], label="root.forward", owner_class="Root", method="forward", calls=2)
    nodes = {
        paths["eager"]: CallNode(
            path=paths["eager"],
            label="root.eager.forward",
            owner_class="Linear",
            method="forward",
            module=shared,
            calls=2,
        ),
        paths["comp"]: CallNode(
            path=paths["comp"],
            label="root.comp.forward",
            owner_class="Seq",
            method="forward",
            module=compiled,
            calls=2,
        ),
        paths["inner"]: CallNode(
            path=paths["inner"],
            label="root.comp.inner.forward",
            owner_class="Linear",
            method="forward",
            module=shared,
            calls=2,
        ),
    }
    tree = CallTree(
        nodes={root.path: root, **nodes},
        iterations=2,
        sequences=[[paths["root"], paths["eager"], paths["comp"], paths["inner"]]] * 2,
        modules=[("root.comp", compiled)],
    )
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:  # only the root runs, at depth 1
        for _ in range(2):
            timer.begin_iteration()
            timer.end_call(timer.begin_call("root.forward", model, "forward", ()))

    text = _warning_text(warning)
    assert "did not run" in text
    assert "eager" in text
    assert "inner" not in text


def test_a_compiled_module_that_did_not_run_is_reported(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing.logger.warning")
    tree, _ = _tree_with_a_compiled_module()

    with TimingRecorder(tree) as timer:
        timer.begin_iteration()

    text = _warning_text(warning)
    assert "did not run" in text
    # only the compiled module's own node is listed; the call inside it is expected to have no time
    assert "child" not in text


def test_more_than_three_untimed_paths_get_an_ellipsis(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing.logger.warning")
    tree = _tree_of([
        [("a",), ("a", "b1"), ("a", "b2"), ("a", "b3"), ("a", "b4")],
        [("a",)],
    ])
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:  # the children never run although a timed depth covers them
        for _ in range(2):  # the schedule times depth 2, then depth 1
            timer.begin_iteration()
            timer.end_call(timer.begin_call("a", model, "forward", ()))

    text = _warning_text(warning)
    assert "did not run in the timing pass" in text
    assert "a > b1, a > b2, a > b3, …" in text
    assert "b4" not in text


def test_a_call_timed_when_its_depth_is_timed_does_not_warn(mocker):
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing.logger.warning")
    root_path, child_path = ("root.forward",), ("root.forward", "root.child.forward")
    tree = _tree_of([[root_path, child_path], [root_path, child_path]])
    model = nn.Linear(2, 2)

    with TimingRecorder(tree) as timer:
        timer.begin_iteration()  # iteration 1 times the root; the child inside the timed call is skipped
        timer.end_call(timer.begin_call("root.forward", model, "forward", ()))
        timer.begin_iteration()  # iteration 2 times the child
        root = timer.begin_call("root.forward", model, "forward", ())
        child = timer.begin_call("root.child.forward", model, "forward", ())
        timer.end_call(child)
        timer.end_call(root)

    assert tree.nodes[("root.forward", "root.child.forward")].time_ms is not None
    warning.assert_not_called()


def test_a_recorder_entering_on_another_thread_fails_while_one_is_active():
    errors = []

    def enter():
        try:
            with TimingRecorder(CallTree()):
                pass
        except RuntimeError as error:
            errors.append(str(error))

    with TimingRecorder(CallTree()) as active:
        thread = threading.Thread(target=enter)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert errors == ["Another call recorder is already recording."]
        assert CallRecorder.active() is active
    assert CallRecorder.active() is None
    assert not CallRecorder._active_lock.locked()


def test_a_stray_exit_does_not_release_the_active_recorder():
    other = TimingRecorder(CallTree())
    with TimingRecorder(CallTree()) as active:
        other.__exit__(None, None, None)
        assert CallRecorder.active() is active
        assert CallRecorder._active_lock.locked()
    assert not CallRecorder._active_lock.locked()


def test_a_transparent_container_gives_its_children_the_same_path_in_both_passes(mocker):
    """``nn.Sequential`` records no node; its children get the same path and time in both passes."""
    mocker.patch("aitune.torch.call_tree.recording.timing.torch.cuda.is_available", return_value=False)
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    x = torch.randn(2, 8)
    structure = StructureRecorder()

    with instrumented(structure, model):
        with structure:
            structure.begin_iteration()
            model(x)
            structure.end_iteration()
        tree = structure.result()
        warning = mocker.patch("aitune.torch.call_tree.recording.timing.logger.warning")
        with TimingRecorder(tree) as timer:
            timer.begin_iteration()
            model(x)
            timer.flush()

    warning.assert_not_called()
    assert ("root.forward",) not in tree.nodes
    assert tree.nodes[("root.0.forward",)].time_ms is not None
