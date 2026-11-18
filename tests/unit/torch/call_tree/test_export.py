# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch

from aitune.torch.call_tree.call_tree import CallTree
from aitune.torch.call_tree.recording import StructureRecorder
from tests.toy_models.call_tree_models import TwoRoleToy, VlaToy, vla_inputs
from tests.unit.torch.call_tree.helpers import instrumented, record


def _object_path(call_path):
    """The derivation the guide documents for readers of the JSON."""
    return call_path.rpartition(" > ")[2].rpartition(".")[0]


def _vla_tree(iterations=2):
    model = VlaToy(steps=2)
    tokens, image = vla_inputs()
    return record(model, lambda: model(tokens, image), iterations=iterations)


def test_to_dict_exports_the_whole_tree():
    tree = _vla_tree()

    d = tree.to_dict()

    assert d["schema_version"] == 1
    assert set(d) == {"schema_version", "iterations", "total_ms", "nodes", "modules", "weights"}
    assert d["iterations"] == tree.iterations
    assert d["total_ms"] == tree.total_ms
    nodes = d["nodes"]
    assert len(nodes) == len(tree.nodes)
    paths = {n["call_path"] for n in nodes}
    for exported, node in zip(nodes, sorted(tree.nodes.values(), key=lambda n: n.first_seen), strict=True):
        assert exported["call_path"] == node.path_str
        assert exported["owner_class"] == node.owner_class
        assert exported["call_path"].rpartition(" > ")[2].rpartition(".")[2] == node.method
        assert _object_path(exported["call_path"]) == node.label.removesuffix(f".{node.method}").removeprefix(
            "root"
        ).removeprefix(".")
        if node.module is not None:
            assert _object_path(exported["call_path"]) in d["modules"]
        assert exported["calls"] == node.calls
        assert exported["time_ms"] == node.time_ms
        assert exported["timings"] == [
            {"iteration": t.iteration, "calls": t.calls, "time_ms": t.time_ms} for t in node.timings
        ]
        assert exported["bytes_in"] == node.bytes_in
        assert exported["bytes_out"] == node.bytes_out
        assert set(exported) == {
            "call_path",
            "owner_class",
            "calls",
            "time_ms",
            "timings",
            "bytes_in",
            "bytes_out",
            "signatures",
            "iteration_starts",
        }
        parent = exported["call_path"].rpartition(" > ")[0]
        assert not parent or parent in paths
    assert nodes[0]["call_path"] == "forward" and _object_path("forward") == ""
    assert _object_path("forward > dit_forward") == ""
    assert _object_path("forward > dit_forward > dit.forward") == "dit"
    assert _object_path("forward > lm_head.forward") == "lm_head"
    json.dumps(d)


def test_to_dict_modules_have_class_and_parameter_bytes():
    tree = _vla_tree()

    modules = tree.to_dict()["modules"]

    dit = tree.nodes[("root.forward", "root.dit_forward", "root.dit.forward")].module
    assert dit is not None
    assert modules["dit"]["class"] == "tests.toy_models.call_tree_models.Dit"
    assert modules["dit"]["parameter_bytes"] == sum(
        t.numel() * t.element_size() for t in (*dit.parameters(), *dit.buffers())
    )
    # VlaToy's lm_head.weight IS llm.embed.weight: parameters() already yields it once
    model = tree.nodes[("root.forward",)].module
    assert model is not None
    tied = sum(t.numel() * t.element_size() for t in (*model.parameters(), *model.buffers()))
    assert modules[""]["parameter_bytes"] == tied


def test_parameter_bytes_count_each_tensor_object_once_across_parameters_and_buffers():
    model = torch.nn.Linear(2, 2, bias=False)  # weight: 4 float32 = 16 bytes
    model.register_buffer("alias", model.weight)  # the same object as a buffer: not counted again
    model.copy = torch.nn.Parameter(model.weight.data)  # another object over the same storage: counted
    x = torch.randn(1, 2)

    tree = record(model, lambda: model(x), iterations=1)

    assert tree.to_dict()["modules"][""]["parameter_bytes"] == 2 * 16


def test_to_dict_modules_parameters_list_own_parameters_and_buffers():
    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 4))
            self.register_buffer("scale", torch.ones(4, dtype=torch.bfloat16))
            self.child = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.child(x) @ self.weight * self.scale

    model = Parent()
    x = torch.randn(1, 4)

    tree = record(model, lambda: model(x), iterations=1)

    modules = tree.to_dict()["modules"]
    assert modules[""]["parameters"] == {
        "weight": {"dtype": "torch.float32", "shape": [4, 4]},
        "scale": {"dtype": "torch.bfloat16", "shape": [4]},
    }
    assert modules["child"]["parameters"] == {
        "weight": {"dtype": "torch.float32", "shape": [4, 4]},
        "bias": {"dtype": "torch.float32", "shape": [4]},
    }


def test_to_dict_modules_parameters_skip_a_missing_bias():
    model = torch.nn.Linear(4, 2, bias=False)
    x = torch.randn(1, 4)

    tree = record(model, lambda: model(x), iterations=1)

    assert tree.to_dict()["modules"][""]["parameters"] == {"weight": {"dtype": "torch.float32", "shape": [2, 4]}}


def test_to_dict_modules_parameters_list_meta_tensors():
    model = torch.nn.Linear(4, 4).to("meta")
    tree = CallTree(iterations=1, modules=[("root", model)])

    assert tree.to_dict()["modules"][""]["parameters"] == {
        "weight": {"dtype": "torch.float32", "shape": [4, 4]},
        "bias": {"dtype": "torch.float32", "shape": [4]},
    }


def test_to_dict_weights_have_the_recorded_accesses_sorted():
    tree = _vla_tree()

    weights = tree.to_dict()["weights"]

    assert list(weights) == sorted(weights)
    assert weights["dit.body.weight"] == [
        {"by": "forward > dit_forward > dit.forward > dit.body.forward", "kind": "read"}
    ]
    reads = weights["lm_head.weight"]
    assert {"by": "forward > lm_head.forward", "kind": "read"} in reads
    for accesses in weights.values():
        assert accesses == sorted(accesses, key=lambda a: (a["kind"], a["by"]))


def test_to_dict_keeps_two_callers_of_one_module_apart():
    model = TwoRoleToy()
    x = torch.randn(4, 8)

    tree = record(model, lambda: model(x), iterations=1)

    nodes = tree.to_dict()["nodes"]
    llm_nodes = [n for n in nodes if _object_path(n["call_path"]) == "llm"]
    assert len(llm_nodes) == 2
    assert {n["call_path"] for n in llm_nodes} == {
        "forward > prefill > llm.forward",
        "forward > generate > llm.forward",
    }


def test_to_dict_exports_signatures_with_exact_shapes():
    class Model(torch.nn.Module):
        def forward(self, x, extra=None):
            return x

    model = Model()
    a, b, c = torch.randn(1, 4, 8), torch.randn(1, 5, 8), torch.randn(8)

    tree = record(model, lambda: (model(a, extra=c), model(b, extra=c)), iterations=1)

    d = tree.to_dict()
    node = next(n for n in d["nodes"] if n["call_path"] == "forward")
    assert set(node["signatures"][0]) == {"call_indices", "inputs", "outputs"}
    assert [sig["call_indices"] for sig in node["signatures"]] == [[0], [1]]
    inputs = node["signatures"][0]["inputs"]
    assert list(inputs["x"]) == ["type", "dtype", "shape"]
    assert inputs["x"] == {"type": "torch.Tensor", "dtype": "torch.float32", "shape": [1, 4, 8]}
    assert inputs["extra"] == {"type": "torch.Tensor", "dtype": "torch.float32", "shape": [8]}
    assert node["signatures"][1]["inputs"]["x"]["shape"] == [1, 5, 8]
    json.dumps(d)


def test_to_dict_exports_object_entries_as_type_only_and_stays_strict_json():
    class Model(torch.nn.Module):
        def forward(self, x, flag=False, cache=None):
            return x

    model = Model()

    tree = record(
        model,
        # fresh tensors: reusing x would make the second call's input the first call's result
        lambda: (
            model(torch.randn(2), flag=True, cache=object()),
            model(torch.randn(2), flag=False, cache=object()),
        ),
        iterations=1,
    )

    signatures = next(n for n in tree.to_dict()["nodes"] if n["call_path"] == "forward")["signatures"]
    inputs = signatures[0]["inputs"]
    assert inputs["cache"] == {"type": "object"}
    assert "flag" not in inputs
    assert signatures[0]["call_indices"] == [0, 1]
    json.dumps(tree.to_dict(), allow_nan=False)


def test_to_dict_exports_signatures_and_iteration_starts():
    class Model(torch.nn.Module):
        def forward(self, x):
            if x.ndim > 2:
                raise ValueError("too many dims")
            return x

    model = Model()

    recorder = StructureRecorder()
    with instrumented(recorder, model), recorder:
        recorder.begin_iteration()
        model(torch.randn(4, 8))
        model(torch.randn(2, 8))
        with pytest.raises(ValueError):
            model(torch.randn(1, 4, 8))
        recorder.end_iteration()

    node = next(n for n in recorder.result().to_dict()["nodes"] if n["call_path"] == "forward")
    assert node["iteration_starts"] == [0]
    signatures = node["signatures"]
    assert [sig["call_indices"] for sig in signatures] == [[0], [1], [2]]
    assert signatures[0]["inputs"]["x"] == {
        "type": "torch.Tensor",
        "dtype": "torch.float32",
        "shape": [4, 8],
    }
    assert signatures[2]["outputs"] is None
    assert signatures[2]["inputs"]["x"]["shape"] == [1, 4, 8]
    json.dumps(recorder.result().to_dict(), allow_nan=False)


def test_to_dict_copies_call_indices():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x

    model = Model()

    tree = record(model, lambda: model(torch.randn(2, 3)), iterations=1)

    tree.to_dict()["nodes"][0]["signatures"][0]["call_indices"].append(99)
    assert tree.nodes[("root.forward",)].signatures[0].call_indices == [0]


def test_to_dict_modules_parameters_list_a_tied_parameter_under_each_name():
    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(2, 2))
            self.alias = self.weight

        def forward(self, x):
            return x @ self.weight

    model = Tied()

    tree = record(model, lambda: model(torch.randn(1, 2)), iterations=1)

    parameters = tree.to_dict()["modules"][""]["parameters"]
    assert set(parameters) == {"weight", "alias"}
