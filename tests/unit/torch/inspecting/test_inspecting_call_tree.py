# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``ait.inspect(..., call_tree=True)``: the recorder pass and its effect on the legacy output."""

import contextlib
import itertools
import unittest.mock

import pytest
import torch
from torch import nn

import aitune.torch.inspecting.inspecting as inspecting_module
from aitune.torch.call_tree import WeightAccess
from aitune.torch.call_tree.recording import StructureRecorder, TimingRecorder
from aitune.torch.distributed import coordinator
from aitune.torch.inspecting import inspect
from aitune.torch.inspecting.module_inspector import ModuleInspector
from tests.toy_models.call_tree_models import PlainPipeline, SequentialSubclassToy, VlaToy, vla_inputs

ITERATIONS = 3


@pytest.fixture
def dataset():
    return torch.randn(10, 8)


def _vla():
    model = VlaToy()
    tokens, image = vla_inputs()
    return model, (lambda _x: model(tokens, image))


def _tree(info):
    assert info.call_tree is not None
    return info.call_tree


def _legacy(info):
    return [(m.object_path, m.execution_count, m.forward_called) for m in info.get_modules()]


def test_default_inspect_records_no_call_tree(dataset):
    model, fn = _vla()

    info = inspect(model, dataset, inference_function=fn, number_of_iterations=ITERATIONS)

    assert info.call_tree is None


def test_call_tree_leaves_legacy_output_unchanged(dataset):
    model, fn = _vla()

    without = inspect(model, dataset, inference_function=fn, number_of_iterations=ITERATIONS, warmup_iterations=1)
    with_graph = inspect(
        model, dataset, inference_function=fn, number_of_iterations=ITERATIONS, warmup_iterations=1, call_tree=True
    )

    assert _legacy(with_graph) == _legacy(without)
    tree = _tree(with_graph)
    assert all(len(node.calls_per_sample) == ITERATIONS for node in tree.nodes.values())
    assert tree.nodes[("root.forward",)].total_calls == ITERATIONS
    assert tree.nodes[("root.forward", "root.dit_forward")].total_calls == 5 * ITERATIONS
    assert ("root.forward", "root.dit_forward", "root.dit.forward") in tree.nodes


def test_default_inspect_keeps_distributed_calls(mocker, dataset):
    model, fn = _vla()
    barrier = mocker.patch("aitune.torch.inspecting.inspecting.coordinator.barrier")
    # inspecting.py and call_tree/sync.py share one coordinator object
    broadcast = mocker.patch(
        "aitune.torch.inspecting.inspecting.coordinator.broadcast_from_rank0", side_effect=lambda value: value
    )

    inspect(model, dataset, inference_function=fn, number_of_iterations=3, warmup_iterations=2)

    assert barrier.call_count == 2 * (3 + 2)
    broadcast.assert_called_once()


def test_call_tree_adds_one_broadcast(mocker, dataset):
    model, fn = _vla()
    barrier = mocker.patch("aitune.torch.inspecting.inspecting.coordinator.barrier")
    broadcast = mocker.patch(
        "aitune.torch.inspecting.inspecting.coordinator.broadcast_from_rank0", side_effect=lambda value: value
    )

    info = inspect(model, dataset, inference_function=fn, number_of_iterations=3, warmup_iterations=2, call_tree=True)

    # warmup, per-module and structure passes run number_of_iterations; the timing pass one inference per planned depth
    assert barrier.call_count == 2 * (2 + 3 + 3 + sum(map(len, TimingRecorder(_tree(info)).depths_per_sample())))
    assert broadcast.call_count == 2
    records = broadcast.call_args_list[1].args[0]
    assert all(len(record.calls_per_sample) == 3 for record in records)
    assert {path for path, *_ in records} == set(_tree(info).nodes)


def test_wrappers_and_apply_are_restored(dataset):
    model, fn = _vla()
    original_apply = nn.Module._apply

    inspect(model, dataset, inference_function=fn, number_of_iterations=2, call_tree=True)

    assert "dit_forward" not in vars(model)
    assert nn.Module._apply is original_apply
    assert model.dit_forward.__func__ is VlaToy.dit_forward


def test_wrappers_and_apply_are_restored_when_inference_fails(dataset):
    model = VlaToy()
    original_apply = nn.Module._apply
    calls = {"n": 0}

    def flaky(_x):
        calls["n"] += 1
        if calls["n"] > 3:
            raise RuntimeError("boom")
        return model(*vla_inputs())

    with pytest.raises(RuntimeError, match="boom"):
        inspect(model, dataset, inference_function=flaky, number_of_iterations=2, warmup_iterations=1, call_tree=True)

    assert "dit_forward" not in vars(model)
    assert nn.Module._apply is original_apply


def test_wrappers_are_restored_when_the_object_walk_fails(mocker, dataset):
    pipe = PlainPipeline()
    # raised in the member loop of the plain root, after its methods are wrapped
    mocker.patch.object(ModuleInspector, "_should_skip_member", side_effect=RuntimeError("walk failed"))

    with pytest.raises(RuntimeError, match="walk failed"):
        inspect(pipe, dataset, inference_function=lambda x: pipe(x), number_of_iterations=2, call_tree=True)

    # the wrapper is set on the instance, shadowing the class method; reset() deletes it
    assert "decode" not in vars(pipe)


class _OptionalStepPipeline(PlainPipeline):
    def __call__(self, x):
        out = self.encoder(x)
        if self.postprocess is not None:
            out = self.postprocess(out)
        return self.decode(out)

    def postprocess(self, x):
        return x


def test_method_disabled_on_the_instance_is_not_wrapped(dataset):
    pipe = _OptionalStepPipeline()
    pipe.postprocess = None  # pyrefly: ignore[bad-assignment]

    info = inspect(pipe, dataset, inference_function=lambda x: pipe(x), number_of_iterations=2, call_tree=True)

    assert pipe.postprocess is None
    assert ("root.postprocess",) not in _tree(info).nodes
    assert ("root.decode",) in _tree(info).nodes


def test_plain_object_methods_and_to_calls_are_recorded(dataset):
    pipe = PlainPipeline()
    info = inspect(pipe, dataset, inference_function=lambda x: pipe(x), number_of_iterations=2, call_tree=True)
    assert ("root.decode",) in _tree(info).nodes
    assert "decode" not in vars(pipe)

    model = SequentialSubclassToy()
    info = inspect(model, dataset, inference_function=lambda x: model.encode(x), number_of_iterations=2, call_tree=True)
    tree = _tree(info)
    assert ("root.encode", "root.forward") in tree.nodes
    assert WeightAccess("encode", "move") in tree.weight_accesses["0.layers.0.proj.weight"]


def test_every_timeable_node_is_timed(mocker, dataset):
    """The timing pass replays each sample once per depth, so two samples already time the whole tree."""
    model, fn = _vla()
    warning = mocker.patch("aitune.torch.call_tree.recording.timing_recorder.logger.warning")

    info = inspect(model, dataset, inference_function=fn, number_of_iterations=2, warmup_iterations=1, call_tree=True)

    tree = _tree(info)
    assert max(len(path) for path in tree.nodes) > 2
    assert all(
        node.total_time_ms is not None
        for node in tree.nodes.values()
        if node.total_calls and not node.inside_compiled_part
    )
    warning.assert_not_called()


def test_the_timing_pass_runs_each_sample_once_per_its_depths(mocker, dataset):
    """``time_replay`` is entered once per depth of ``TimingRecorder.depths_per_sample()``, in order."""
    model, fn = _vla()
    begin = mocker.spy(TimingRecorder, "time_replay")

    info = inspect(model, dataset, inference_function=fn, number_of_iterations=2, warmup_iterations=1, call_tree=True)

    tree = _tree(info)
    assert [c.args[1] for c in begin.call_args_list] == list(
        itertools.chain.from_iterable(TimingRecorder(tree).depths_per_sample())
    )


def test_ranks_verify_the_same_timing_depths(mocker, dataset):
    model, fn = _vla()
    verify = mocker.spy(coordinator, "verify_equal")

    info = inspect(model, dataset, inference_function=fn, number_of_iterations=2, warmup_iterations=1, call_tree=True)

    verify.assert_called_once_with(TimingRecorder(_tree(info)).depths_per_sample(), "call tree timing depths")


def test_calls_that_only_some_samples_make_are_timed(mocker):
    class Branch(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x):
            return self.lin(x)

    class Conditional(nn.Module):
        def __init__(self):
            super().__init__()
            self.always = nn.Linear(8, 8)
            self.sometimes = Branch()

        def forward(self, x):
            out = self.always(x)
            if x.flatten()[0] > 0:
                out = out + self.sometimes(x)
            return out

    model = Conditional()
    dataset = [torch.ones(8), -torch.ones(8), torch.ones(8), -torch.ones(8)]
    warning = mocker.patch("aitune.torch.call_tree.recording.timing_recorder.logger.warning")

    info = inspect(
        model,
        dataset,
        inference_function=lambda x: model(x),
        number_of_iterations=4,
        warmup_iterations=1,
        call_tree=True,
    )

    tree = _tree(info)
    branch = ("root.forward", "root.sometimes.forward")
    deep = (*branch, "root.sometimes.lin.forward")
    assert tree.nodes[branch].calls_per_sample == [1, 0, 1, 0]
    # only the branch reaches depth 3, so only samples 0 and 2 get a replay at that depth
    assert TimingRecorder(tree).depths_per_sample() == [[1, 2, 3], [1, 2], [1, 2, 3], [1, 2]]
    assert tree.nodes[branch].total_time_ms is not None
    assert tree.nodes[deep].total_time_ms is not None
    warning.assert_not_called()


def test_a_part_compiled_with_module_compile_logs_no_timing_warning(mocker, dataset):
    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x):
            return torch.relu(self.lin(x))

    class Root(nn.Module):
        def __init__(self, child):
            super().__init__()
            self.child = child

        def forward(self, x):
            return self.child(x)

    inner = Inner()
    inner.compile(backend="eager")
    model = Root(inner)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing_recorder.logger.warning")

    info = inspect(
        model,
        dataset,
        inference_function=lambda x: model(x),
        number_of_iterations=4,
        warmup_iterations=1,
        call_tree=True,
    )

    assert ("root.forward", "root.child.forward", "root.child.lin.forward") in _tree(info).nodes
    warning.assert_not_called()


def test_a_part_compiled_through_its_forward_logs_no_timing_warning(mocker, dataset):
    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x):
            return torch.relu(self.lin(x))

    class Root(nn.Module):
        def __init__(self, child):
            super().__init__()
            self.child = child

        def forward(self, x):
            return self.child(x)

    inner = Inner()
    inner.forward = torch.compile(inner.forward, backend="eager")
    model = Root(inner)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing_recorder.logger.warning")

    info = inspect(
        model,
        dataset,
        inference_function=lambda x: model(x),
        number_of_iterations=4,
        warmup_iterations=1,
        call_tree=True,
    )

    assert ("root.forward", "root.child.forward", "root.child.lin.forward") in _tree(info).nodes
    assert _tree(info).nodes[("root.forward", "root.child.forward")].total_time_ms is not None
    warning.assert_not_called()


def test_a_compiled_sequential_gets_a_timed_node_and_its_children_stay_untimed(mocker, dataset):
    """``nn.Module.compile()`` on an exact ``nn.Sequential``: the container is a unit of its own — recorded and
    timed as one call — and the calls inside the compiled part get no time without a warning."""

    class Root(nn.Module):
        def __init__(self, child):
            super().__init__()
            self.child = child

        def forward(self, x):
            return self.child(x)

    child = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
    child.compile(backend="eager")
    model = Root(child)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing_recorder.logger.warning")

    info = inspect(
        model,
        dataset,
        inference_function=lambda x: model(x),
        number_of_iterations=4,
        warmup_iterations=1,
        call_tree=True,
    )

    tree = _tree(info)
    node = tree.nodes[("root.forward", "root.child.forward")]
    assert node.total_time_ms is not None
    inner = tree.nodes[("root.forward", "root.child.forward", "root.child.0.forward")]
    assert inner.total_time_ms is None
    warning.assert_not_called()


def test_a_sequential_with_a_compiled_forward_gets_a_timed_node_and_its_children_stay_untimed(mocker, dataset):
    """A ``torch.compile``'d ``forward`` on an exact ``nn.Sequential`` makes the container a unit of its own, like
    ``nn.Module.compile()``: its call is timed as one node, and the calls inside are marked as inside it."""

    class Root(nn.Module):
        def __init__(self, child):
            super().__init__()
            self.child = child

        def forward(self, x):
            return self.child(x)

    child = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
    child.forward = torch.compile(child.forward, backend="eager")
    model = Root(child)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing_recorder.logger.warning")

    info = inspect(
        model,
        dataset,
        inference_function=lambda x: model(x),
        number_of_iterations=4,
        warmup_iterations=1,
        call_tree=True,
    )

    tree = _tree(info)
    node = tree.nodes[("root.forward", "root.child.forward")]
    assert node.total_time_ms is not None
    assert not node.inside_compiled_part
    inner = tree.nodes[("root.forward", "root.child.forward", "root.child.0.forward")]
    assert inner.total_time_ms is None
    assert inner.inside_compiled_part
    warning.assert_not_called()


def test_a_compiled_sequential_times_its_own_node(mocker, dataset):
    """The compiled part's own node is timeable: the plan covers its depth, while the depths that only exist
    inside it are left out and never warned about."""

    class Root(nn.Module):
        def __init__(self, child):
            super().__init__()
            self.child = child

        def forward(self, x):
            return self.child(x)

    child = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
    child.compile(backend="eager")
    model = Root(child)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing_recorder.logger.warning")

    info = inspect(
        model,
        dataset,
        inference_function=lambda x: model(x),
        number_of_iterations=1,
        warmup_iterations=1,
        call_tree=True,
    )

    tree = _tree(info)
    own = tree.nodes[("root.forward", "root.child.forward")]
    assert own.inside_compiled_part is False
    for path in ("root.child.0.forward", "root.child.1.forward", "root.child.2.forward"):
        node = tree.nodes[("root.forward", "root.child.forward", path)]
        assert node.inside_compiled_part is True
        assert node.total_time_ms is None
    assert own.total_time_ms is not None
    warning.assert_not_called()


@pytest.mark.parametrize("how", ["module.compile", "forward"])
def test_a_compiled_part_times_its_own_node(mocker, dataset, how):
    """The plan covers the timeable depths, so the compiled part's call is timed and nothing warns."""

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x):
            return torch.relu(self.lin(x))

    class Root(nn.Module):
        def __init__(self, child):
            super().__init__()
            self.child = child

        def forward(self, x):
            return self.child(x)

    inner = Inner()
    if how == "module.compile":
        inner.compile(backend="eager")
    else:
        inner.forward = torch.compile(inner.forward, backend="eager")
    model = Root(inner)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing_recorder.logger.warning")

    info = inspect(
        model,
        dataset,
        inference_function=lambda x: model(x),
        number_of_iterations=2,
        warmup_iterations=1,
        call_tree=True,
    )

    tree = _tree(info)
    assert tree.nodes[("root.forward", "root.child.forward")].total_time_ms is not None
    warning.assert_not_called()


def test_a_disabled_call_inside_a_compiled_part(mocker, dataset):
    """Inside a compiled part, a ``torch.compiler.disable``'d call and everything below it are marked inside:
    the calls run eagerly, but the timing pass never sees the disabled call, so it cannot match them to their
    nodes and cannot time them."""

    class Disabled(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        @torch.compiler.disable
        def forward(self, x):
            return self.lin(x)

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.off = Disabled()

        def forward(self, x):
            return self.off(x)

    class Root(nn.Module):
        def __init__(self, child):
            super().__init__()
            self.child = child

        def forward(self, x):
            return self.child(x)

    inner = Inner()
    inner.compile(backend="eager")
    model = Root(inner)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing_recorder.logger.warning")

    info = inspect(
        model,
        dataset,
        inference_function=lambda x: model(x),
        number_of_iterations=4,
        warmup_iterations=1,
        call_tree=True,
    )

    tree = _tree(info)
    disabled = tree.nodes[("root.forward", "root.child.forward", "root.child.off.forward")]
    assert disabled.inside_compiled_part is True
    assert disabled.total_time_ms is None
    lin = tree.nodes[("root.forward", "root.child.forward", "root.child.off.forward", "root.child.off.lin.forward")]
    assert lin.inside_compiled_part is True
    assert lin.total_time_ms is None
    warning.assert_not_called()


def test_a_torch_compiled_sequential_gets_a_timed_node(mocker, dataset):
    """``torch.compile(nn.Sequential)``: the ``OptimizedModule`` gets the compiled call's node — recorded under its
    label and timed — while the calls inside stay untimed without a warning."""

    class Root(nn.Module):
        def __init__(self, child):
            super().__init__()
            self.child = child

        def forward(self, x):
            return self.child(x)

    torch._dynamo.reset()
    model = Root(torch.compile(nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8)), backend="eager"))
    warning = mocker.patch("aitune.torch.call_tree.recording.timing_recorder.logger.warning")

    info = inspect(
        model,
        dataset,
        inference_function=lambda x: model(x),
        number_of_iterations=4,
        warmup_iterations=1,
        call_tree=True,
    )
    torch._dynamo.reset()

    tree = _tree(info)
    node = tree.nodes[("root.forward", "root.child.forward")]
    assert node.total_time_ms is not None
    inner = tree.nodes[("root.forward", "root.child.forward", "root.child._orig_mod.0.forward")]
    assert inner.total_time_ms is None
    warning.assert_not_called()


def test_a_compiled_modules_encode_call_exposes_its_children(mocker, dataset):
    """``module.compile()`` compiles the call (``forward``) only: an ``encode()`` of the compiled module runs
    eagerly in the timing pass, so the calls below it are seen and timed like any other — no warning."""

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x):
            return torch.relu(self.lin(x))

        def encode(self, x):
            return self.lin(x)

    class Root(nn.Module):
        def __init__(self, child):
            super().__init__()
            self.child = child

        def forward(self, x):
            return self.child.encode(x)

    inner = Inner()
    inner.compile(backend="eager")
    model = Root(inner)
    warning = mocker.patch("aitune.torch.call_tree.recording.timing_recorder.logger.warning")

    info = inspect(
        model,
        dataset,
        inference_function=lambda x: model(x),
        number_of_iterations=4,
        warmup_iterations=1,
        call_tree=True,
    )

    tree = _tree(info)
    path = ("root.forward", "root.child.encode", "root.child.lin.forward")
    assert path in tree.nodes
    assert tree.nodes[path].total_time_ms is not None
    warning.assert_not_called()


def test_a_torch_compiled_child_that_recurses_records_the_nested_call(mocker, dataset):
    """A compiled module whose forward calls itself: the mark suppresses only the first entry, so the
    structure pass records the nested call and the timing pass reports it, not as a missing call."""

    class Recursive(nn.Module):
        def forward(self, x, depth=0):
            if depth == 0:
                return self.forward(x, depth + 1)
            return x.relu()

    class Root(nn.Module):
        def __init__(self, child):
            super().__init__()
            self.child = child

        def forward(self, x):
            return self.child(x)

    torch._dynamo.reset()
    model = Root(torch.compile(Recursive(), backend="eager"))
    warning = mocker.patch("aitune.torch.call_tree.recording.timing_recorder.logger.warning")

    info = inspect(
        model,
        dataset,
        inference_function=lambda x: model(x),
        number_of_iterations=4,
        warmup_iterations=1,
        call_tree=True,
    )
    torch._dynamo.reset()

    path = ("root.forward", "root.child.forward", "root.child._orig_mod.forward")
    assert path in _tree(info).nodes
    warning.assert_not_called()


def test_timing_pass_ends_every_run(mocker, dataset):
    model, fn = _vla()
    spy = mocker.spy(TimingRecorder, "_resolve_events")

    info = inspect(model, dataset, inference_function=fn, number_of_iterations=3, warmup_iterations=2, call_tree=True)

    # once per timing replay, plus one when the timing pass ends
    assert spy.call_count == sum(map(len, TimingRecorder(_tree(info)).depths_per_sample())) + 1


def test_default_inspect_creates_no_recorder(mocker, dataset):
    model, fn = _vla()
    structure_spy = mocker.spy(StructureRecorder, "__init__")
    timing_spy = mocker.spy(TimingRecorder, "__init__")

    inspect(model, dataset, inference_function=fn, number_of_iterations=2, warmup_iterations=1)

    assert structure_spy.call_count == 0
    assert timing_spy.call_count == 0


def test_forward_is_not_recorded_while_compiling(mocker, dataset):
    model, fn = _vla()
    without = inspect(model, dataset, inference_function=fn, number_of_iterations=ITERATIONS, warmup_iterations=1)

    real_record_call_tree = inspecting_module._record_call_tree

    def record_while_compiling(*args, **kwargs):
        with unittest.mock.patch.object(torch.compiler, "is_compiling", return_value=True):
            return real_record_call_tree(*args, **kwargs)

    mocker.patch("aitune.torch.inspecting.inspecting._record_call_tree", side_effect=record_while_compiling)
    warning = mocker.patch("aitune.torch.inspecting.inspecting.logger.warning")
    replay = mocker.spy(TimingRecorder, "time_replay")

    info = inspect(
        model, dataset, inference_function=fn, number_of_iterations=ITERATIONS, warmup_iterations=1, call_tree=True
    )

    assert _tree(info).nodes == {}
    assert _legacy(info) == _legacy(without)
    assert "recorded no calls" in warning.call_args.args[0]
    replay.assert_not_called()


def test_the_timing_pass_is_skipped_when_no_node_is_timeable(mocker, dataset):
    """When the replay plan is empty for every sample, the pass replays nothing and logs the skip."""
    model, fn = _vla()
    mocker.patch.object(TimingRecorder, "_is_timeable", return_value=False)
    replay = mocker.spy(TimingRecorder, "time_replay")
    info_log = mocker.spy(inspecting_module.logger, "info")

    info = inspect(
        model, dataset, inference_function=fn, number_of_iterations=ITERATIONS, warmup_iterations=1, call_tree=True
    )

    assert _tree(info).nodes
    replay.assert_not_called()
    assert "timing pass is skipped" in str(info_log.call_args_list)


def test_event_resolution_failure_is_raised_through_the_rank_failure_context(mocker, dataset):
    model, fn = _vla()
    spy = mocker.spy(coordinator, "raise_if_any_rank_fails")

    def fail_in_the_timing_pass(self):
        raise RuntimeError("bad events")

    mocker.patch.object(TimingRecorder, "_resolve_events", new=fail_in_the_timing_pass)
    original_apply = nn.Module._apply

    with pytest.raises(RuntimeError, match="bad events"):
        inspect(model, dataset, inference_function=fn, number_of_iterations=2, warmup_iterations=1, call_tree=True)

    spy.assert_any_call("Timing a call tree replay")
    assert nn.Module._apply is original_apply


def test_timing_pass_entry_failure_goes_through_the_rank_failure_context(mocker, dataset):
    model, fn = _vla()
    spy = mocker.spy(coordinator, "raise_if_any_rank_fails")
    begin = mocker.spy(TimingRecorder, "time_replay")
    mocker.patch.object(TimingRecorder, "_start", side_effect=RuntimeError("bad start"))
    original_apply = nn.Module._apply

    with pytest.raises(RuntimeError, match="bad start"):
        inspect(model, dataset, inference_function=fn, number_of_iterations=2, warmup_iterations=1, call_tree=True)

    spy.assert_any_call("Starting the call tree timing pass")
    begin.assert_not_called()  # the failure on entry ran no timed inference
    assert nn.Module._apply is original_apply


def test_timing_pass_entry_peer_failure_exits_the_recorder(mocker, dataset):
    model, fn = _vla()
    real_guard = coordinator.raise_if_any_rank_fails
    begin = mocker.spy(TimingRecorder, "time_replay")
    stop = mocker.spy(TimingRecorder, "_stop")
    original_apply = nn.Module._apply

    @contextlib.contextmanager
    def _peer_failed():
        yield
        raise RuntimeError("peer failed")

    mocker.patch.object(
        coordinator,
        "raise_if_any_rank_fails",
        side_effect=lambda op: _peer_failed() if op == "Starting the call tree timing pass" else real_guard(op),
    )

    with pytest.raises(RuntimeError, match="peer failed"):
        inspect(model, dataset, inference_function=fn, number_of_iterations=2, warmup_iterations=1, call_tree=True)

    begin.assert_not_called()  # entry succeeded locally but no timed inference ran
    stop.assert_called_once()
    assert stop.call_args.kwargs["completed"] is False  # the recorder exited through the failed path
    assert nn.Module._apply is original_apply


def test_timing_pass_exit_failure_goes_through_the_rank_failure_context(mocker, dataset):
    model, fn = _vla()
    spy = mocker.spy(coordinator, "raise_if_any_rank_fails")
    mocker.patch.object(TimingRecorder, "_stop", side_effect=RuntimeError("bad stop"))
    original_apply = nn.Module._apply

    with pytest.raises(RuntimeError, match="bad stop"):
        inspect(model, dataset, inference_function=fn, number_of_iterations=2, warmup_iterations=1, call_tree=True)

    spy.assert_any_call("Ending the call tree timing pass")
    assert nn.Module._apply is original_apply


def test_structure_result_failure_goes_through_the_rank_failure_context(mocker, dataset):
    model, fn = _vla()
    spy = mocker.spy(coordinator, "raise_if_any_rank_fails")
    mocker.patch.object(StructureRecorder, "result", side_effect=RuntimeError("bad result"))
    original_apply = nn.Module._apply

    with pytest.raises(RuntimeError, match="bad result"):
        inspect(model, dataset, inference_function=fn, number_of_iterations=2, warmup_iterations=1, call_tree=True)

    spy.assert_any_call("Reading the call tree's structure")
    assert nn.Module._apply is original_apply


def test_sample_description_failure_goes_through_the_rank_failure_context(mocker, dataset):
    model, fn = _vla()
    spy = mocker.spy(coordinator, "raise_if_any_rank_fails")
    mocker.patch(
        "aitune.torch.call_tree.recording.structure_recorder.describe_sample_inputs",
        side_effect=RuntimeError("bad description"),
    )
    original_apply = nn.Module._apply

    with pytest.raises(RuntimeError, match="bad description"):
        inspect(model, dataset, inference_function=fn, number_of_iterations=2, warmup_iterations=1, call_tree=True)

    spy.assert_any_call("Describing a call tree sample")
    assert nn.Module._apply is original_apply


def test_compiled_child_inspects_and_its_calls_appear_in_the_tree(dataset):
    """A ``torch.compile`` child must not crash the legacy pass; the call tree sees inside it (force_eager)."""

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x):
            return torch.relu(self.lin(x))

    class Root(nn.Module):
        def __init__(self, child):
            super().__init__()
            self.child = child

        def forward(self, x):
            return self.child(x)

    model = Root(torch.compile(Inner(), backend="eager"))
    fn = lambda x: model(x)  # noqa: E731

    info = inspect(model, dataset, inference_function=fn, number_of_iterations=ITERATIONS, warmup_iterations=1)
    assert info.call_tree is None

    info = inspect(
        model, dataset, inference_function=fn, number_of_iterations=ITERATIONS, warmup_iterations=1, call_tree=True
    )
    labels = [n.label for n in _tree(info).nodes.values()]
    assert any("child" in label for label in labels)
    assert any("lin" in label for label in labels)


def test_torch_compiled_child_is_one_node_and_resolves_for_shared_weights(dataset):
    """``torch.compile(module)``: the ``OptimizedModule``'s ``forward`` is the node; ``_orig_mod`` adds none."""
    from typing import cast

    from aitune.torch.call_tree import shared_weights_of

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x):
            return torch.relu(self.lin(x))

    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.child = torch.compile(Inner(), backend="eager")

        def forward(self, x):
            return self.child(x)

    torch._dynamo.reset()
    model = Root()
    info = inspect(
        model,
        dataset,
        inference_function=lambda x: model(x),
        number_of_iterations=ITERATIONS,
        warmup_iterations=1,
        call_tree=True,
    )

    paths = {n.path_str: n.total_calls for n in _tree(info).nodes.values()}
    assert paths == {
        "forward": ITERATIONS,
        "forward > child.forward": ITERATIONS,
        "forward > child.forward > child._orig_mod.lin.forward": ITERATIONS,
    }
    assert shared_weights_of(_tree(info), cast(nn.Module, model.child)) == []


def test_module_compiled_child_is_recorded_once_and_counts_stay_unchanged(dataset):
    """A child compiled with ``nn.Module.compile()`` gets its nodes; the structure pass leaves counts alone."""

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x):
            return torch.relu(self.lin(x))

    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.child = Inner()

        def forward(self, x):
            return self.child(x)

    torch._dynamo.reset()
    model = Root()
    model.child.compile(backend="eager")
    fn = lambda x: model(x)  # noqa: E731
    kwargs = {"inference_function": fn, "number_of_iterations": ITERATIONS, "warmup_iterations": 1, "min_depth": 1}

    without_tree = _legacy(inspect(model, dataset, **kwargs))
    info = inspect(model, dataset, call_tree=True, **kwargs)

    assert _legacy(info) == without_tree
    nodes = _tree(info).nodes
    assert nodes[("root.forward", "root.child.forward")].total_calls == ITERATIONS
    assert ("root.forward", "root.child.forward", "root.child.lin.forward") in nodes


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)

    def forward(self, x):
        return self.proj(x)


class _TunedBlocksModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])
        self.head = nn.Linear(8, 8)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def test_call_tree_after_tune_matches_eager_call_paths():
    """A wrapped, tuned module keeps its call path; the wrapper's own methods are not recorded."""
    from aitune.torch.backend import TorchEagerBackend
    from aitune.torch.inspecting.wrapping import wrap
    from aitune.torch.module.wrapper_module import Module, ModuleState
    from aitune.torch.module_registry import MODULE_REGISTRY
    from aitune.torch.tune_strategy import OneBackendStrategy
    from aitune.torch.tuning import tune

    def call_paths(info):
        return {n.path_str for n in info.call_tree.nodes.values()}

    model = _TunedBlocksModel()
    x = torch.randn(4, 8)

    eager_info = inspect(model, x, number_of_iterations=1, warmup_iterations=0, min_depth=2, call_tree=True)
    eager_paths = call_paths(eager_info)
    assert "forward > blocks.0.forward" in eager_paths

    blocks0 = [m for m in eager_info.get_modules() if m.object_path == ".blocks.0"]
    wrap(model, blocks0, strategy=OneBackendStrategy(backend=TorchEagerBackend()))
    tune(model, x, batch_sizes=[1], device="cpu")
    tuned_block = model.blocks[0]
    assert isinstance(tuned_block, Module) and tuned_block.state == ModuleState.TUNED

    expected = model(x.unsqueeze(0))
    tuned_info = inspect(model, x, number_of_iterations=1, warmup_iterations=0, min_depth=2, call_tree=True)
    tuned_paths = call_paths(tuned_info)

    assert tuned_paths == eager_paths
    assert all("_proxy_forward" not in p and "_restore_original_forward" not in p for p in tuned_paths)
    assert all("._" not in p.rpartition(" > ")[2] for p in tuned_paths)

    # Inspect restores the wrapper's proxy forward; the model still runs through the tuned module.
    assert torch.equal(model(x.unsqueeze(0)), expected)
    assert isinstance(model.blocks[0], Module) and model.blocks[0].state == ModuleState.TUNED
    assert MODULE_REGISTRY.modules["blocks.0"] is model.blocks[0]

    again = inspect(model, x, number_of_iterations=1, warmup_iterations=0, min_depth=2, call_tree=True)
    assert call_paths(again) == eager_paths


def test_max_timing_depth_leaves_deep_nodes_untimed_without_a_warning(mocker, dataset):
    """Nodes deeper than the limit are still recorded — calls, bytes, weight accesses — just not timed."""
    model, fn = _vla()
    warning = mocker.patch("aitune.torch.call_tree.recording.timing_recorder.logger.warning")

    info = inspect(
        model,
        dataset,
        inference_function=fn,
        number_of_iterations=2,
        warmup_iterations=1,
        call_tree=True,
        max_timing_depth=2,
    )

    tree = _tree(info)
    assert max(len(path) for path in tree.nodes) > 2  # the structure pass still records the whole tree
    deep = [node for node in tree.nodes.values() if len(node.path) > 2]
    assert all(node.total_calls > 0 for node in deep)
    assert all(node.total_time_ms is None for node in deep)
    shallow = [node for node in tree.nodes.values() if node.total_calls and len(node.path) <= 2]
    assert shallow and all(node.total_time_ms is not None for node in shallow)
    warning.assert_not_called()


@pytest.mark.parametrize("bad", [-1, -5])
def test_max_timing_depth_rejects_negative_values(bad, dataset):
    model, fn = _vla()

    with pytest.raises(ValueError, match="max_timing_depth must be a non-negative int or None"):
        inspect(model, dataset, inference_function=fn, call_tree=True, max_timing_depth=bad)


def test_max_timing_depth_zero_skips_the_timing_pass(mocker, dataset):
    """``max_timing_depth=0`` runs only the structure pass: the tree is complete, every ``total_time_ms`` stays None."""
    model, fn = _vla()
    timing = mocker.patch("aitune.torch.inspecting.inspecting._time_call_tree")

    info = inspect(
        model,
        dataset,
        inference_function=fn,
        number_of_iterations=2,
        warmup_iterations=1,
        call_tree=True,
        max_timing_depth=0,
    )

    timing.assert_not_called()
    tree = _tree(info)
    assert max(len(path) for path in tree.nodes) > 2  # the structure pass still records the whole tree
    assert all(node.total_calls > 0 for node in tree.nodes.values())
    assert all(node.total_time_ms is None for node in tree.nodes.values())


def test_times_are_reported_per_sample():
    """Two samples of different shapes give the root two signatures, each with its own timed total."""
    model = nn.Linear(8, 8)
    samples = [torch.randn(1, 8), torch.randn(3, 8)]

    info = inspect(model, samples, number_of_iterations=2, warmup_iterations=0, call_tree=True)

    tree = _tree(info)
    timed = [node for node in tree.nodes.values() if node.total_time_ms is not None]
    assert timed
    for node in timed:
        assert node.time_ms_per_sample is not None
        assert len(node.time_ms_per_sample) == 2
        assert sum(node.time_ms_per_sample) == pytest.approx(node.total_time_ms)


def test_max_timing_depth_zero_leaves_all_detailed_times_none(dataset):
    model, fn = _vla()

    info = inspect(
        model,
        dataset,
        inference_function=fn,
        number_of_iterations=2,
        warmup_iterations=1,
        call_tree=True,
        max_timing_depth=0,
    )

    tree = _tree(info)
    assert all(node.total_time_ms is None for node in tree.nodes.values())
    assert all(node.time_ms_per_sample is None for node in tree.nodes.values())


def test_samples_are_recorded_with_the_dataset_wraparound():
    """``tree.samples[i]`` describes sample ``i``: with more iterations than samples, the dataset restarts."""
    model = nn.Linear(8, 8)
    s0, s1 = torch.randn(1, 8), torch.randn(3, 8)

    info = inspect(model, [s0, s1], number_of_iterations=3, warmup_iterations=0, call_tree=True)

    tree = _tree(info)
    assert [tuple(sample.inputs["args[0]"].shape) for sample in tree.samples] == [
        tuple(s0.unsqueeze(0).shape),
        tuple(s1.unsqueeze(0).shape),
        tuple(s0.unsqueeze(0).shape),
    ]
    assert all(len(node.calls_per_sample) == 3 for node in tree.nodes.values())
    assert all(len(node.time_ms_per_sample) == 3 for node in tree.nodes.values() if node.time_ms_per_sample)
