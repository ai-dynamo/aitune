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
    total_calls: int
    calls_per_sample: list[int]
    time_ms_per_sample: list[float] | None
    total_bytes_in: int
    total_bytes_out: int
    signatures: list[CallSignature]

    def structure(self) -> tuple[str, int, list[int]]:
        """The part every rank must have recorded identically."""
        return self.owner_class, self.total_calls, self.calls_per_sample


def synchronize_call_tree(tree: CallTree) -> CallTree:
    """Apply rank zero's call counts, call times, byte totals and recorded signatures to the local call tree.

    Updates ``tree`` in place and returns it.

    Every rank must have recorded the same tree: the same call paths (adapter names), with the same owner class,
    call count and ``CallNode.calls_per_sample`` per node, so each node's calls fall into the same samples on every
    rank. Rank zero's measured times, tensor bytes and per-node signatures replace the local ones, so every rank
    reports the same values; the times are written into the local ``samples``. The bytes and signatures are not
    compared, because tensor-parallel ranks can hold differently sized shards. ``.to()`` callers and recorded weight
    accesses stay local: they follow from the verified structure, so rank zero's signature call indices map to the
    same samples on every rank. Each sample's ``inputs`` stay local too, as the arguments the rank received.

    Args:
        tree: The local call tree.

    Returns:
        The same tree with rank zero's call times, byte totals and signatures.

    Raises:
        RuntimeError: If the local tree differs from rank zero's.
    """
    local = tuple(
        _NodeRecord(
            path,
            node.owner_class,
            node.total_calls,
            list(node.calls_per_sample),
            node.time_ms_per_sample,
            node.total_bytes_in,
            node.total_bytes_out,
            node.signatures,
        )
        for path, node in tree.nodes.items()
    )
    rank_zero_records = coordinator.broadcast_from_rank0(local)

    with coordinator.raise_if_any_rank_fails("Synchronizing call tree"):
        rank_zero_paths = {record.path for record in rank_zero_records}
        if set(tree.nodes) != rank_zero_paths:
            missing = sorted(rank_zero_paths - set(tree.nodes))
            unexpected = sorted(set(tree.nodes) - rank_zero_paths)
            raise RuntimeError(
                f"Local call tree differs from rank zero: missing={missing!r}, unexpected={unexpected!r}"
            )
        for record in rank_zero_records:
            node = tree.nodes[record.path]
            local_structure = (node.owner_class, node.total_calls, node.calls_per_sample)
            if local_structure != record.structure():
                raise RuntimeError(
                    f"Local call {record.path[-1]!r} differs from rank zero: "
                    f"(owner class, total calls, calls per sample) local={local_structure!r}, "
                    f"rank zero={record.structure()!r}"
                )
    for record in rank_zero_records:
        node = tree.nodes[record.path]
        if record.time_ms_per_sample is None:
            for sample in tree.samples:
                sample.time_ms.pop(record.path, None)
        else:
            for i, ms in enumerate(record.time_ms_per_sample):
                tree.samples[i].time_ms[record.path] = ms
        node.total_bytes_in = record.total_bytes_in
        node.total_bytes_out = record.total_bytes_out
        node.signatures = list(record.signatures)
    return tree
