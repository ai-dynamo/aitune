# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aitune.torch.call_tree.sync import synchronize_call_tree
from tests.unit.torch.call_tree.helpers import make_tree

BROADCAST = "aitune.torch.call_tree.sync.coordinator.broadcast_from_rank0"


def _tree(time_ms: float = 1.0, calls: int = 1):
    return make_tree({"root.forward": (1, 10.0, {"root.head.forward": (calls, time_ms, {})})}, iterations=2)


def test_rank_zero_times_are_applied(mocker):
    tree = _tree(time_ms=1.0)
    rank_zero = _tree(time_ms=4.0)
    records = tuple((k, n.owner_class, n.calls, n.time_ms, n.bytes_in, n.bytes_out) for k, n in rank_zero.nodes.items())
    broadcast = mocker.patch(BROADCAST, return_value=(2, records))

    synchronize_call_tree(tree)

    broadcast.assert_called_once()
    assert tree.nodes[("root.forward", "root.head.forward")].time_ms == 4.0


def test_different_keys_raise(mocker):
    tree = _tree()
    records = (
        (("root.forward",), "root", 1, 10.0, 0, 0),
        (("root.forward", "root.other.forward"), "other", 1, 1.0, 0, 0),
    )
    mocker.patch(BROADCAST, return_value=(2, records))

    with pytest.raises(RuntimeError, match="differs from rank zero"):
        synchronize_call_tree(tree)


def test_different_call_counts_raise(mocker):
    tree = _tree(calls=1)
    rank_zero = _tree(calls=3)
    records = tuple((k, n.owner_class, n.calls, n.time_ms, n.bytes_in, n.bytes_out) for k, n in rank_zero.nodes.items())
    mocker.patch(BROADCAST, return_value=(2, records))

    with pytest.raises(RuntimeError, match="head.forward"):
        synchronize_call_tree(tree)


def test_rank_zero_bytes_are_applied(mocker):
    tree = _tree(time_ms=1.0)
    rank_zero = _tree(time_ms=1.0)
    for node in rank_zero.nodes.values():
        node.bytes_in, node.bytes_out = 100, 200
    records = tuple((k, n.owner_class, n.calls, n.time_ms, n.bytes_in, n.bytes_out) for k, n in rank_zero.nodes.items())
    mocker.patch(BROADCAST, return_value=(2, records))

    synchronize_call_tree(tree)

    head = tree.nodes[("root.forward", "root.head.forward")]
    assert (head.bytes_in, head.bytes_out) == (100, 200)
