# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``ait.inspect(..., call_tree=True)``: the recorder pass and its effect on the legacy output."""

import unittest.mock

import pytest
import torch
from torch import nn

import aitune.torch.inspecting.inspecting as inspecting_module
from aitune.torch.call_tree import WeightAccess
from aitune.torch.call_tree.recorder import CallRecorder
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
    assert tree.iterations == ITERATIONS
    assert tree.nodes[("root.forward",)].calls == ITERATIONS
    assert tree.nodes[("root.forward", "root.dit_forward")].calls == 5 * ITERATIONS
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

    assert barrier.call_count == 2 * (3 + 2 + 3)
    assert broadcast.call_count == 2
    iterations, records = broadcast.call_args_list[1].args[0]
    assert iterations == 3
    assert {key for key, *_ in records} == set(_tree(info).nodes)


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


def test_recording_pass_flushes_after_every_inference(mocker, dataset):
    model, fn = _vla()
    spy = mocker.spy(CallRecorder, "flush")

    inspect(model, dataset, inference_function=fn, number_of_iterations=3, warmup_iterations=2, call_tree=True)

    # once per recording iteration, plus the one in result()
    assert spy.call_count == 3 + 1


def test_default_inspect_creates_no_recorder(mocker, dataset):
    model, fn = _vla()
    spy = mocker.spy(CallRecorder, "__init__")

    inspect(model, dataset, inference_function=fn, number_of_iterations=2, warmup_iterations=1)

    assert spy.call_count == 0


def test_forward_is_not_recorded_while_compiling(mocker, dataset):
    model, fn = _vla()
    without = inspect(model, dataset, inference_function=fn, number_of_iterations=ITERATIONS, warmup_iterations=1)

    real_record_call_tree = inspecting_module._record_call_tree

    def record_while_compiling(*args, **kwargs):
        with unittest.mock.patch.object(torch.compiler, "is_compiling", return_value=True):
            return real_record_call_tree(*args, **kwargs)

    mocker.patch("aitune.torch.inspecting.inspecting._record_call_tree", side_effect=record_while_compiling)

    info = inspect(
        model, dataset, inference_function=fn, number_of_iterations=ITERATIONS, warmup_iterations=1, call_tree=True
    )

    assert _tree(info).nodes == {}
    assert _legacy(info) == _legacy(without)


def test_flush_failure_is_raised_through_the_rank_failure_context(mocker, dataset):
    model, fn = _vla()
    spy = mocker.spy(coordinator, "raise_if_any_rank_fails")
    mocker.patch.object(CallRecorder, "flush", side_effect=RuntimeError("bad events"))
    original_apply = nn.Module._apply

    with pytest.raises(RuntimeError, match="bad events"):
        inspect(model, dataset, inference_function=fn, number_of_iterations=2, warmup_iterations=1, call_tree=True)

    spy.assert_any_call("Resolving call tree timings")
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


def test_module_compiled_child_is_recorded_once_and_counts_stay_unchanged(dataset):
    """A child compiled with ``nn.Module.compile()`` gets its call-tree nodes, and the recording pass leaves counts alone."""

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
    assert nodes[("root.forward", "root.child.forward")].calls == ITERATIONS
    assert ("root.forward", "root.child.forward", "root.child.lin.forward") in nodes
