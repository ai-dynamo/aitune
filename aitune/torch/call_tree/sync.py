# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synchronize a recorded call tree across distributed ranks."""

from typing import NamedTuple

from aitune.torch.call_tree.call_tree import CallTree
from aitune.torch.call_tree.nodes import CallPath, CallSignature
from aitune.torch.distributed import coordinator


class _NodeRecord(NamedTuple):
    """What rank zero broadcasts for one node: the structure every rank must share, and the values it applies."""

    path: CallPath
    owner_class: str
    calls: int
    calls_per_iteration: list[int]
    time_ms: float | None
    bytes_in: int
    bytes_out: int
    signatures: list[CallSignature]

    def structure(self) -> tuple[str, int, list[int]]:
        """The part every rank must have recorded identically."""
        return self.owner_class, self.calls, self.calls_per_iteration


def synchronize_call_tree(tree: CallTree) -> CallTree:
    """Apply rank zero's call counts, call times, byte totals and recorded signatures to the local call tree.

    Updates ``tree`` in place and returns it.

    Every rank must have recorded the same tree: the same call paths (adapter names), with the same owner class,
    call count and ``CallTree.calls_per_iteration`` per node, so each node's calls fall into the same iterations on
    every rank. The measured times, tensor bytes and the per-node signatures are taken from rank zero,
    so every rank reports the same values; the bytes and signatures are not compared, because tensor-parallel
    ranks can hold differently sized shards. Sequences, ``.to()`` callers and recorded weight accesses stay
    local; they follow from the verified structure, so rank zero's signature call indices map to the same
    iterations on every rank.

    Args:
        tree: The local call tree.

    Returns:
        The same tree with rank zero's call times, byte totals and signatures.

    Raises:
        RuntimeError: If the local tree differs from rank zero's.
    """
    calls_per_iteration = tree.calls_per_iteration()
    local = (
        tree.iterations,
        tuple(
            _NodeRecord(
                path,
                node.owner_class,
                node.calls,
                calls_per_iteration[path],
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
        rank_zero_paths = {record.path for record in rank_zero_records}
        if set(tree.nodes) != rank_zero_paths or tree.iterations != rank_zero_iterations:
            missing = sorted(rank_zero_paths - set(tree.nodes))
            unexpected = sorted(set(tree.nodes) - rank_zero_paths)
            raise RuntimeError(
                f"Local call tree differs from rank zero: missing={missing!r}, unexpected={unexpected!r}, "
                f"iterations local={tree.iterations} rank zero={rank_zero_iterations}"
            )
        for record in rank_zero_records:
            node = tree.nodes[record.path]
            local_structure = (node.owner_class, node.calls, calls_per_iteration[record.path])
            if local_structure != record.structure():
                raise RuntimeError(
                    f"Local call {record.path[-1]!r} differs from rank zero: "
                    f"(owner class, calls, calls per iteration) local={local_structure!r}, "
                    f"rank zero={record.structure()!r}"
                )
    for record in rank_zero_records:
        node = tree.nodes[record.path]
        node.time_ms = record.time_ms
        node.bytes_in = record.bytes_in
        node.bytes_out = record.bytes_out
        node.signatures = list(record.signatures)
    return tree
