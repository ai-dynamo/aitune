# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from aitune.torch.call_tree.nodes import TensorLeaf
from tests.unit.torch.call_tree.helpers import record

ROOT = ("root.forward",)
HEAD = ("root.forward", "root.head.forward")


class _Head(nn.Module):
    def forward(self, x):
        return x + 1


class _Delegating(nn.Module):
    """Passes its own argument to a child and returns the child's result."""

    def __init__(self):
        super().__init__()
        self.head = _Head()

    def forward(self, x):
        return self.head(x)


def _outputs(tree, path):
    outputs = tree.nodes[path].signatures[0].outputs
    assert outputs is not None
    return outputs


def test_a_caller_argument_is_the_source_of_a_child_input():
    model = _Delegating()

    tree = record(model, lambda: model(torch.randn(4)), iterations=1)

    assert tree.nodes[HEAD].signatures[0].inputs["x"].source == (ROOT, "inputs", "x")


def test_a_child_result_is_the_source_of_the_callers_output():
    model = _Delegating()

    tree = record(model, lambda: model(torch.randn(4)), iterations=1)

    assert _outputs(tree, ROOT)["output"].source == (HEAD, "outputs", "output")


class _Siblings(nn.Module):
    """Passes the result of one child to a later child."""

    def __init__(self):
        super().__init__()
        self.a = _Head()
        self.b = _Head()

    def forward(self, x):
        return self.b(self.a(x))


def test_a_sibling_result_is_the_source_of_a_later_siblings_input():
    model = _Siblings()
    a_path = ("root.forward", "root.a.forward")

    tree = record(model, lambda: model(torch.randn(4)), iterations=1)

    assert tree.nodes[("root.forward", "root.b.forward")].signatures[0].inputs["x"].source == (
        a_path,
        "outputs",
        "output",
    )


class _Chain(nn.Module):
    """Calls the same child twice, feeding its first result back in, like a cached step."""

    def __init__(self):
        super().__init__()
        self.head = _Head()

    def forward(self, x):
        return self.head(self.head(x))


def test_a_nodes_previous_result_is_the_source_of_a_later_calls_input():
    model = _Chain()

    tree = record(model, lambda: model(torch.randn(4)), iterations=1)

    signatures = tree.nodes[HEAD].signatures
    assert [s.call_indices for s in signatures] == [[0], [1]]
    assert signatures[0].inputs["x"].source == (ROOT, "inputs", "x")
    assert signatures[1].inputs["x"].source == (HEAD, "outputs", "output")


class _Identity(nn.Module):
    def forward(self, x):
        return x


def test_an_argument_returned_unchanged_is_the_source_of_the_output():
    model = _Identity()

    tree = record(model, lambda: model(torch.randn(4)), iterations=1)

    assert _outputs(tree, ROOT)["output"].source == (ROOT, "inputs", "x")


def test_the_side_tells_a_parameter_named_output_from_the_result():
    class Passthrough(nn.Module):
        def forward(self, output):
            return output

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = Passthrough()

        def forward(self, x):
            return self.head(x)

    model = Model()

    tree = record(model, lambda: model(torch.randn(4)), iterations=1)

    # the head returns its argument ``output``: its input and its result are both named ``output``
    assert _outputs(tree, HEAD)["output"].source == (HEAD, "inputs", "output")
    assert _outputs(tree, ROOT)["output"].source == (HEAD, "outputs", "output")


class _Computing(nn.Module):
    """Makes a value in its own code, which therefore has no source."""

    def __init__(self):
        super().__init__()
        self.head = _Head()

    def forward(self, x):
        return self.head(x * 2)


def test_a_value_made_inside_the_call_has_no_source():
    model = _Computing()

    tree = record(model, lambda: model(torch.randn(4)), iterations=1)

    assert tree.nodes[HEAD].signatures[0].inputs["x"].source is None


class _DictResult(nn.Module):
    def forward(self, x):
        return {"y": x + 1}


class _DictArg(nn.Module):
    def forward(self, inputs):
        return inputs["y"]


class _Nested(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = _DictResult()
        self.b = _DictArg()

    def forward(self, x):
        return self.b(self.a(x))


def test_sources_keep_nested_container_names_on_both_ends():
    model = _Nested()
    a_path = ("root.forward", "root.a.forward")
    b_path = ("root.forward", "root.b.forward")

    tree = record(model, lambda: model(torch.randn(4)), iterations=1)

    assert tree.nodes[b_path].signatures[0].inputs["inputs['y']"].source == (a_path, "outputs", "output['y']")
    assert _outputs(tree, ROOT)["output"].source == (b_path, "outputs", "output")


class _Cache:
    pass


class _SlotsCache:
    __slots__ = ("x",)


class _ObjectArg(nn.Module):
    def forward(self, x, cache):
        return x + 1


class _Passing(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = _ObjectArg()

    def forward(self, x, cache):
        return self.head(x, cache)


def test_a_non_tensor_object_is_linked():
    model, cache = _Passing(), _Cache()

    tree = record(model, lambda: model(torch.randn(4), cache), iterations=1)

    assert tree.nodes[HEAD].signatures[0].inputs["cache"].source == (ROOT, "inputs", "cache")


def test_an_object_without_weakref_support_gets_no_source():
    model, cache = _Passing(), _SlotsCache()

    tree = record(model, lambda: model(torch.randn(4), cache), iterations=1)

    assert tree.nodes[HEAD].signatures[0].inputs["cache"].source is None


class _Stashing(nn.Module):
    """Stashes its child's result; the next sample's call with it sees no scope it lives in."""

    def __init__(self):
        super().__init__()
        self.head = _Head()
        self.cached = None

    def forward(self, x):
        r = self.head(x)
        if self.cached is not None:
            self.head(self.cached)
        self.cached = r
        return r


def test_an_outermost_calls_result_is_the_source_of_a_later_outermost_call():
    # like a generate loop: the inference function passes one call's result to the next call
    model = _Head()

    def step():
        model(model(torch.randn(4)))

    tree = record(model, step, iterations=1)

    signatures = tree.nodes[ROOT].signatures
    assert [s.call_indices for s in signatures] == [[0], [1]]
    assert signatures[0].inputs["x"].source is None
    assert signatures[1].inputs["x"].source == (ROOT, "outputs", "output")


def test_an_outermost_calls_result_is_not_linked_into_the_next_sample():
    model = _Head()
    previous = [torch.randn(4)]

    def step():
        previous[0] = model(previous[0])

    tree = record(model, step, iterations=2)

    assert [s.inputs["x"].source for s in tree.nodes[ROOT].signatures] == [None]


def test_nothing_is_linked_across_iterations():
    model = _Stashing()

    tree = record(model, lambda: model(torch.randn(4)), iterations=2)

    signatures = {tuple(s.call_indices): s for s in tree.nodes[HEAD].signatures}
    assert signatures[(2,)].inputs["x"].source is None
    assert signatures[(0, 1)].inputs["x"].source == (ROOT, "inputs", "x")


class _Loop(nn.Module):
    """Iterates the child on a state it made itself first, then on the child's results."""

    def __init__(self):
        super().__init__()
        self.head = _Head()

    def forward(self, x, steps):
        state = x * 2
        for _ in range(steps):
            state = self.head(state)
        return state


def test_a_loop_callers_made_value_and_previous_results_give_two_signatures():
    model = _Loop()

    tree = record(model, lambda: model(torch.randn(4), 3), iterations=1)

    signatures = tree.nodes[HEAD].signatures
    assert [s.call_indices for s in signatures] == [[0], [1, 2]]
    assert signatures[0].inputs["x"].source is None
    assert signatures[1].inputs["x"].source == (HEAD, "outputs", "output")
    first, later = signatures[0].inputs["x"], signatures[1].inputs["x"]
    assert isinstance(first, TensorLeaf) and isinstance(later, TensorLeaf)
    assert first.shape == later.shape


def test_to_dict_exports_sources():
    model = _Delegating()
    tree = record(model, lambda: model(torch.randn(4)), iterations=1)

    exported = tree.to_dict()

    nodes = {n["call_path"]: n for n in exported["nodes"]}
    assert nodes["forward > head.forward"]["signatures"][0]["inputs"]["x"]["source"] == {
        "call_path": "forward",
        "side": "inputs",
        "name": "x",
    }
    assert nodes["forward"]["signatures"][0]["outputs"]["output"]["source"] == {
        "call_path": "forward > head.forward",
        "side": "outputs",
        "name": "output",
    }
    assert "source" not in nodes["forward"]["signatures"][0]["inputs"]["x"]
