# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import torch

from tests.toy_models.call_tree_models import TwoRoleToy, VlaToy, vla_inputs
from tests.unit.torch.call_tree.helpers import record


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
        assert exported["call_path"] == node.path
        assert exported["owner_class"] == node.owner_class
        assert exported["call_path"].rpartition(" > ")[2].rpartition(".")[2] == node.method
        assert _object_path(exported["call_path"]) == node.label.removesuffix(f".{node.method}").removeprefix(
            "root"
        ).removeprefix(".")
        if node.module is not None:
            assert _object_path(exported["call_path"]) in d["modules"]
        assert exported["calls"] == node.calls
        assert exported["time_ms"] == node.time_ms
        assert exported["bytes_in"] == node.bytes_in
        assert exported["bytes_out"] == node.bytes_out
        assert set(exported) == {
            "call_path",
            "owner_class",
            "calls",
            "time_ms",
            "bytes_in",
            "bytes_out",
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


def test_save_writes_the_same_json(tmp_path):
    tree = _vla_tree()

    out = tmp_path / "t.json"
    assert tree.save(out) is None
    assert json.loads(out.read_text()) == tree.to_dict()


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
