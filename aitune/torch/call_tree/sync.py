# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synchronize a recorded call tree across distributed ranks."""

from aitune.torch.call_tree.call_tree import CallTree
from aitune.torch.distributed import coordinator


def synchronize_call_tree(tree: CallTree) -> CallTree:
    """Apply rank zero's call counts and timings to the local call tree.

    Every rank must have recorded the same tree: the same node keys (adapter names), with the same owner class and
    call count per key. Times are taken from rank zero so that every rank reports the same times. Sequences, ``.to()`` callers
    and recorded weight accesses stay local; they follow from the verified structure.

    Args:
        tree: The local call tree.

    Returns:
        The same tree with rank zero's times.

    Raises:
        RuntimeError: If the local tree differs from rank zero's.
    """
    local = (
        tree.iterations,
        tuple((key, node.owner_class, node.calls, node.time_ms) for key, node in tree.nodes.items()),
    )
    rank_zero_iterations, rank_zero_records = coordinator.broadcast_from_rank0(local)

    with coordinator.raise_if_any_rank_fails("Synchronizing call tree"):
        rank_zero_keys = {key for key, _, _, _ in rank_zero_records}
        if set(tree.nodes) != rank_zero_keys or tree.iterations != rank_zero_iterations:
            missing = sorted(rank_zero_keys - set(tree.nodes))
            unexpected = sorted(set(tree.nodes) - rank_zero_keys)
            raise RuntimeError(
                f"Local call tree differs from rank zero: missing={missing!r}, unexpected={unexpected!r}, "
                f"iterations local={tree.iterations} rank zero={rank_zero_iterations}"
            )
        for key, owner_class, calls, time_ms in rank_zero_records:
            node = tree.nodes[key]
            if (node.owner_class, node.calls) != (owner_class, calls):
                raise RuntimeError(
                    f"Local call {key[-1]!r} differs from rank zero: "
                    f"local={(node.owner_class, node.calls)!r}, rank zero={(owner_class, calls)!r}"
                )
            node.time_ms = time_ms
    return tree
