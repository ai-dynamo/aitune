# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from aitune.torch.call_tree.call_tree import CallTree
from aitune.torch.call_tree.nodes import CallNode
from aitune.torch.call_tree.sync import _NodeRecord, synchronize_call_tree
from tests.unit.torch.call_tree.helpers import record

BROADCAST = "aitune.torch.call_tree.sync.coordinator.broadcast_from_rank0"


def _tree(total_time_ms=1.0, calls: int = 1):
    head_path = ("root.forward", "root.head.forward")
    root = CallNode(
        path=("root.forward",),
        label="root.forward",
        owner_class="root",
        method="forward",
        calls=1,
        total_time_ms=10.0,
        first_seen=0,
        children=[head_path],
    )
    head = CallNode(
        path=head_path,
        label="root.head.forward",
        owner_class="head",
        method="forward",
        calls=calls,
        total_time_ms=total_time_ms,
        first_seen=1,
    )
    tree = CallTree(nodes={root.path: root, head_path: head})
    tree.sequences = [[], []]
    return tree


def _records(tree):
    counts = tree.calls_per_sample()
    return tuple(
        _NodeRecord(
            k,
            n.owner_class,
            n.calls,
            counts[k],
            n.total_time_ms,
            n.total_bytes_in,
            n.total_bytes_out,
            n.signatures,
        )
        for k, n in tree.nodes.items()
    )


def test_rank_zero_times_are_applied(mocker):
    tree = _tree(total_time_ms=0.5)
    rank_zero = _tree(total_time_ms=4.0)
    records = _records(rank_zero)
    broadcast = mocker.patch(BROADCAST, return_value=(2, records))

    synchronize_call_tree(tree)

    broadcast.assert_called_once()
    head = tree.nodes[("root.forward", "root.head.forward")]
    assert head.total_time_ms == 4.0


def test_different_keys_raise(mocker):
    tree = _tree()
    records = (
        _NodeRecord(("root.forward",), "root", 1, [], None, 0, 0, []),
        _NodeRecord(("root.forward", "root.other.forward"), "other", 1, [], None, 0, 0, []),
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


def test_different_calls_per_sample_raise(mocker):
    """Equal call totals are not enough: rank zero's call indices must fall into the same samples."""
    tree, rank_zero = _tree(calls=2), _tree(calls=2)
    outer, head = ("root.forward",), ("root.forward", "root.head.forward")
    tree.sequences = [[outer, head, head], [outer]]
    rank_zero.sequences = [[outer, head], [outer, head]]
    mocker.patch(BROADCAST, return_value=(2, _records(rank_zero)))

    with pytest.raises(RuntimeError, match="head.forward"):
        synchronize_call_tree(tree)


def test_a_mismatch_leaves_the_local_tree_unchanged(mocker):
    tree, rank_zero = _tree(calls=1), _tree(calls=3)
    rank_zero.nodes[("root.forward",)].total_time_ms = 99.0  # matches and comes first
    mocker.patch(BROADCAST, return_value=(2, _records(rank_zero)))

    with pytest.raises(RuntimeError, match="head.forward"):
        synchronize_call_tree(tree)

    assert tree.nodes[("root.forward",)].total_time_ms == 10.0


def test_rank_zero_bytes_are_applied(mocker):
    tree = _tree()
    rank_zero = _tree()
    for node in rank_zero.nodes.values():
        node.total_bytes_in, node.total_bytes_out = 100, 200
    records = _records(rank_zero)
    mocker.patch(BROADCAST, return_value=(2, records))

    synchronize_call_tree(tree)

    head = tree.nodes[("root.forward", "root.head.forward")]
    assert (head.total_bytes_in, head.total_bytes_out) == (100, 200)


def test_rank_zero_shapes_are_applied(mocker):
    """Rank zero's shapes are applied, not compared: tensor-parallel shards can differ in shape."""
    local_model, rank_zero_model = torch.nn.Linear(2, 2), torch.nn.Linear(4, 2)
    tree = record(local_model, lambda: local_model(torch.randn(1, 2)), num_samples=1)
    rank_zero = record(rank_zero_model, lambda: rank_zero_model(torch.randn(1, 4)), num_samples=1)
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
