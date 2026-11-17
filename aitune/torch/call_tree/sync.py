# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synchronize a recorded call tree across distributed ranks."""

from aitune.torch.call_tree.call_tree import CallTree
from aitune.torch.distributed import coordinator


def synchronize_call_tree(tree: CallTree) -> CallTree:
    """Apply rank zero's call counts, timings, byte totals and recorded signatures to the local call tree.

    Every rank must have recorded the same tree: the same call paths (adapter names), with the same owner class,
    call count and ``CallTree.iteration_starts`` per node, so each node's calls fall into the same iterations on
    every rank. Times, tensor bytes and the per-node signatures are taken from rank zero so that every rank reports
    the same values; the bytes and signatures are not compared, because tensor-parallel ranks can hold differently
    sized shards. Sequences, ``.to()`` callers and recorded weight accesses stay local; they follow from the
    verified structure, so rank zero's signature call indices map to the same iterations on every rank.

    Args:
        tree: The local call tree.

    Returns:
        The same tree with rank zero's times, byte totals and signatures.

    Raises:
        RuntimeError: If the local tree differs from rank zero's.
    """
    iteration_starts = tree.iteration_starts()
    local = (
        tree.iterations,
        tuple(
            (
                path,
                node.owner_class,
                node.calls,
                iteration_starts[path],
                node.time_ms,
                node.bytes_in,
                node.bytes_out,
                node.signatures,
            )
            for path, node in tree.nodes.items()
        ),
    )
    rank_zero_iterations, rank_zero_records = coordinator.broadcast_from_rank0(local)

    with coordinator.raise_if_any_rank_fails("Synchronizing call tree"):
        rank_zero_paths = {path for path, *_ in rank_zero_records}
        if set(tree.nodes) != rank_zero_paths or tree.iterations != rank_zero_iterations:
            missing = sorted(rank_zero_paths - set(tree.nodes))
            unexpected = sorted(set(tree.nodes) - rank_zero_paths)
            raise RuntimeError(
                f"Local call tree differs from rank zero: missing={missing!r}, unexpected={unexpected!r}, "
                f"iterations local={tree.iterations} rank zero={rank_zero_iterations}"
            )
        for path, owner_class, calls, starts, *_ in rank_zero_records:
            node = tree.nodes[path]
            local_structure = (node.owner_class, node.calls, iteration_starts[path])
            if local_structure != (owner_class, calls, starts):
                raise RuntimeError(
                    f"Local call {path[-1]!r} differs from rank zero: (owner class, calls, iteration starts) "
                    f"local={local_structure!r}, rank zero={(owner_class, calls, starts)!r}"
                )
    for path, _, _, _, time_ms, bytes_in, bytes_out, signatures in rank_zero_records:
        node = tree.nodes[path]
        node.time_ms = time_ms
        node.bytes_in = bytes_in
        node.bytes_out = bytes_out
        node.signatures = list(signatures)
    return tree
