# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The same model recorded through the AOT adapter and a test-only JIT adapter gives the same call tree.

Node names differ by design (object paths vs class and creation index); the structure must not. The toy model
includes an exact ``nn.Sequential`` (skipped in both modes) and leaves out what the modes see differently by design:
methods on plain objects, modules created before JIT patching, and modules created inside ``forward``.
"""

import torch

from aitune.torch.call_tree.call_tree import CallNode, CallTree
from aitune.torch.call_tree.recorder import CallRecorder
from aitune.torch.inspecting import inspect
from tests.toy_models.call_tree_models import VlaToy, vla_inputs
from tests.unit.torch.call_tree.jit_adapter import jit_recording_adapter

ITERATIONS = 2


def _canonical(tree: CallTree, node: CallNode) -> tuple:
    return (node.owner_class, node.method, node.calls, tuple(_canonical(tree, tree.nodes[k]) for k in node.children))


def _structure(tree: CallTree) -> list[tuple]:
    return [_canonical(tree, root) for root in tree.roots]


def _aot_tree(tokens, image) -> CallTree:
    model = VlaToy()
    info = inspect(
        model,
        torch.zeros(4, 1),
        inference_function=lambda _x: model(tokens, image),
        number_of_iterations=ITERATIONS,
        warmup_iterations=1,
        call_tree=True,
    )
    assert info.call_tree is not None
    return info.call_tree


def _jit_tree(tokens, image) -> CallTree:
    recorder = CallRecorder()
    with jit_recording_adapter(recorder):
        model = VlaToy()
        model(tokens, image)  # warmup, not recorded
        with recorder.recording(), torch.no_grad():
            for _ in range(ITERATIONS):
                recorder.begin_iteration()
                model(tokens, image)
                recorder.end_iteration()
    return recorder.result()


def test_aot_and_jit_record_the_same_call_tree():
    tokens, image = vla_inputs()

    aot, jit = _aot_tree(tokens, image), _jit_tree(tokens, image)

    assert _structure(aot) == _structure(jit)
    assert len(aot.nodes) == len(jit.nodes)
    assert not any(node.owner_class == "Sequential" for node in jit.nodes.values())
    # names differ by design
    assert ("root.forward",) in aot.nodes
    assert all(key[0].startswith("VlaToy#") for key in jit.nodes)
