# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from aitune.torch.call_tree.nodes import CallNode, Leaf, TensorLeaf
from aitune.torch.call_tree.sync import _NodeRecord, synchronize_call_tree
from tests.unit.torch.call_tree.helpers import record, tree_of

BROADCAST = "aitune.torch.call_tree.sync.coordinator.broadcast_from_rank0"


def _tree(total_time_ms=1.0, total_calls: int = 1):
    head_path = ("root.forward", "root.head.forward")
    root = CallNode(
        path=("root.forward",),
        label="root.forward",
        owner_class="root",
        method="forward",
        total_time_ms=10.0,
        first_seen=0,
        children=[head_path],
    )
    head = CallNode(
        path=head_path,
        label="root.head.forward",
        owner_class="head",
        method="forward",
        total_time_ms=total_time_ms,
        first_seen=1,
    )
    return tree_of([root, head], calls_per_sample={root.path: [1, 0], head_path: [total_calls, 0]})


def _records(tree):
    return tuple(
        _NodeRecord(
            k,
            n.owner_class,
            n.total_calls,
            n.calls_per_sample,
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
    broadcast = mocker.patch(BROADCAST, return_value=records)

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
    mocker.patch(BROADCAST, return_value=records)

    with pytest.raises(RuntimeError, match="differs from rank zero"):
        synchronize_call_tree(tree)


def test_different_call_counts_raise(mocker):
    tree = _tree(total_calls=1)
    rank_zero = _tree(total_calls=3)
    records = _records(rank_zero)
    mocker.patch(BROADCAST, return_value=records)

    with pytest.raises(RuntimeError, match="head.forward"):
        synchronize_call_tree(tree)


def test_different_calls_per_sample_raise(mocker):
    """Equal call totals are not enough: rank zero's call indices must fall into the same samples."""
    tree, rank_zero = _tree(total_calls=2), _tree(total_calls=2)
    head = ("root.forward", "root.head.forward")
    tree.samples[0].calls[head] = 2
    rank_zero.samples[0].calls[head] = 1
    rank_zero.samples[1].calls[head] = 1
    mocker.patch(BROADCAST, return_value=_records(rank_zero))

    with pytest.raises(RuntimeError, match="head.forward"):
        synchronize_call_tree(tree)


def test_a_mismatch_leaves_the_local_tree_unchanged(mocker):
    tree, rank_zero = _tree(total_calls=1), _tree(total_calls=3)
    rank_zero.nodes[("root.forward",)].total_time_ms = 99.0  # matches and comes first
    mocker.patch(BROADCAST, return_value=_records(rank_zero))

    with pytest.raises(RuntimeError, match="head.forward"):
        synchronize_call_tree(tree)

    assert tree.nodes[("root.forward",)].total_time_ms == 10.0


def test_rank_zero_bytes_are_applied(mocker):
    tree = _tree()
    rank_zero = _tree()
    for node in rank_zero.nodes.values():
        node.total_bytes_in, node.total_bytes_out = 100, 200
    records = _records(rank_zero)
    mocker.patch(BROADCAST, return_value=records)

    synchronize_call_tree(tree)

    head = tree.nodes[("root.forward", "root.head.forward")]
    assert (head.total_bytes_in, head.total_bytes_out) == (100, 200)


def test_rank_zero_shapes_are_applied(mocker):
    """Rank zero's shapes are applied, not compared: tensor-parallel shards can differ in shape."""
    local_model, rank_zero_model = torch.nn.Linear(2, 2), torch.nn.Linear(4, 2)
    tree = record(local_model, lambda: local_model(torch.randn(1, 2)), num_samples=1)
    rank_zero = record(rank_zero_model, lambda: rank_zero_model(torch.randn(1, 4)), num_samples=1)
    for tree_ in (tree, rank_zero):
        for sample in tree_.samples:
            sample.calls = dict.fromkeys(tree_.nodes, 1)
        for node in tree_.nodes.values():
            node.owner_class = "Same"
    rank_zero_signatures = [n.signatures for n in rank_zero.nodes.values()]
    assert all(rank_zero_signatures)
    assert [n.signatures for n in tree.nodes.values()] != rank_zero_signatures
    mocker.patch(BROADCAST, return_value=_records(rank_zero))

    synchronize_call_tree(tree)

    assert [n.signatures for n in tree.nodes.values()] == rank_zero_signatures


def test_samples_are_not_synchronized(mocker):
    """Each rank keeps its own recorded samples; sync only applies rank zero's per-node values."""
    tree, rank_zero = _tree(), _tree()
    inputs: dict[str, Leaf] = {"args[0]": TensorLeaf("args[0]", "torch.Tensor", torch.float32, (2, 2))}
    tree.samples[0].inputs = inputs
    mocker.patch(BROADCAST, return_value=_records(rank_zero))

    synchronize_call_tree(tree)

    assert len(tree.samples) == 2
    assert tree.samples[0].inputs == inputs
