# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``ait.inspect(..., call_tree=True)`` on the GPU: the recorded tree, its CUDA times and the unchanged legacy output."""

import pytest
import torch

import aitune.torch as ait
from tests.functional_ci.models import seeded
from tests.toy_models.call_tree_models import VlaToy, vla_inputs

ITERATIONS = 3


def _vla(device: torch.device):
    with seeded(device, 0):
        model = VlaToy().to(device).eval()
    tokens, image = (t.to(device) for t in vla_inputs())
    return model, (lambda _x: model(tokens, image)), torch.randn(10, 8, device=device)


def _inspect(model, fn, dataset, call_tree: bool):
    with torch.no_grad():
        return ait.inspect(
            model,
            dataset,
            inference_function=fn,
            number_of_iterations=ITERATIONS,
            warmup_iterations=1,
            call_tree=call_tree,
        )


@pytest.mark.matrix("inspecting.call-tree/aot")
def test_call_tree_records_the_calls_and_their_cuda_times(device: torch.device):
    """The tree has the model's calls with per-iteration counts, and every recorded call has a CUDA time."""
    model, fn, dataset = _vla(device)

    tree = _inspect(model, fn, dataset, call_tree=True).call_tree

    assert tree is not None
    assert tree.iterations == ITERATIONS
    assert tree.nodes[("root.forward",)].calls == ITERATIONS
    assert tree.nodes[("root.forward", "root.dit_forward")].calls == 5 * ITERATIONS
    assert ("root.forward", "root.dit_forward", "root.dit.forward") in tree.nodes
    assert all(node.time_ms > 0 for node in tree.nodes.values())


@pytest.mark.matrix("inspecting.call-tree-modules-unchanged/aot")
def test_call_tree_leaves_the_selected_modules_unchanged(device: torch.device):
    """``get_modules()`` selects the same modules with the same counts with and without ``call_tree=True``."""
    model, fn, dataset = _vla(device)

    without = _inspect(model, fn, dataset, call_tree=False)
    with_tree = _inspect(model, fn, dataset, call_tree=True)

    def legacy(info):
        return [(m.object_path, m.execution_count, m.forward_called) for m in info.get_modules()]

    assert legacy(with_tree) == legacy(without)
