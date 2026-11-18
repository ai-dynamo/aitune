# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from aitune.torch.call_tree.sync import _NodeRecord, synchronize_call_tree
from tests.unit.torch.call_tree.helpers import make_tree, record, timed

BROADCAST = "aitune.torch.call_tree.sync.coordinator.broadcast_from_rank0"


def _tree(time_ms: float = 1.0, calls: int = 1):
    return make_tree({"root.forward": (1, 10.0, {"root.head.forward": (calls, time_ms, {})})}, iterations=2)


def _records(tree):
    starts = tree.iteration_starts()
    return tuple(
        _NodeRecord(k, n.owner_class, n.calls, starts[k], n.timings, n.bytes_in, n.bytes_out, n.signatures)
        for k, n in tree.nodes.items()
    )


def test_rank_zero_times_are_applied(mocker):
    tree = _tree(time_ms=1.0)
    rank_zero = _tree(time_ms=4.0)
    records = _records(rank_zero)
    broadcast = mocker.patch(BROADCAST, return_value=(2, records))

    synchronize_call_tree(tree)

    broadcast.assert_called_once()
    assert tree.nodes[("root.forward", "root.head.forward")].time_ms == 4.0


def test_different_keys_raise(mocker):
    tree = _tree()
    records = (
        _NodeRecord(("root.forward",), "root", 1, [], [], 0, 0, []),
        _NodeRecord(("root.forward", "root.other.forward"), "other", 1, [], [], 0, 0, []),
    )
    mocker.patch(BROADCAST, return_value=(2, records))

    with pytest.raises(RuntimeError, match="differs from rank zero"):
        synchronize_call_tree(tree)


def test_different_call_counts_raise(mocker):
    tree = _tree(calls=1)
    rank_zero = _tree(calls=3)
    records = _records(rank_zero)
    mocker.patch(BROADCAST, return_value=(2, records))

    with pytest.raises(RuntimeError, match="head.forward"):
        synchronize_call_tree(tree)


def test_different_calls_per_iteration_raise(mocker):
    """Equal call totals are not enough: rank zero's call indices must fall into the same iterations."""
    tree, rank_zero = _tree(calls=2), _tree(calls=2)
    outer, head = ("root.forward",), ("root.forward", "root.head.forward")
    tree.sequences = [[outer, head, head], [outer]]
    rank_zero.sequences = [[outer, head], [outer, head]]
    mocker.patch(BROADCAST, return_value=(2, _records(rank_zero)))

    with pytest.raises(RuntimeError, match="head.forward"):
        synchronize_call_tree(tree)


def test_a_mismatch_leaves_the_local_tree_unchanged(mocker):
    tree, rank_zero = _tree(time_ms=1.0, calls=1), _tree(time_ms=1.0, calls=3)
    rank_zero.nodes[("root.forward",)].timings = timed(99.0, calls=1)  # the outer node matches and comes first
    mocker.patch(BROADCAST, return_value=(2, _records(rank_zero)))

    with pytest.raises(RuntimeError, match="head.forward"):
        synchronize_call_tree(tree)

    assert tree.nodes[("root.forward",)].time_ms == 10.0


def test_rank_zero_bytes_are_applied(mocker):
    tree = _tree(time_ms=1.0)
    rank_zero = _tree(time_ms=1.0)
    for node in rank_zero.nodes.values():
        node.bytes_in, node.bytes_out = 100, 200
    records = _records(rank_zero)
    mocker.patch(BROADCAST, return_value=(2, records))

    synchronize_call_tree(tree)

    head = tree.nodes[("root.forward", "root.head.forward")]
    assert (head.bytes_in, head.bytes_out) == (100, 200)


def test_rank_zero_shapes_are_applied(mocker):
    """Rank zero's shapes are applied, not compared: tensor-parallel shards can differ in shape."""
    local_model, rank_zero_model = torch.nn.Linear(2, 2), torch.nn.Linear(4, 2)
    tree = record(local_model, lambda: local_model(torch.randn(1, 2)), iterations=1)
    rank_zero = record(rank_zero_model, lambda: rank_zero_model(torch.randn(1, 4)), iterations=1)
    for tree_ in (tree, rank_zero):
        for node in tree_.nodes.values():
            node.calls = 1
            node.owner_class = "Same"
    rank_zero_signatures = [n.signatures for n in rank_zero.nodes.values()]
    assert all(rank_zero_signatures)
    assert [n.signatures for n in tree.nodes.values()] != rank_zero_signatures
    mocker.patch(BROADCAST, return_value=(1, _records(rank_zero)))

    synchronize_call_tree(tree)

    assert [n.signatures for n in tree.nodes.values()] == rank_zero_signatures
