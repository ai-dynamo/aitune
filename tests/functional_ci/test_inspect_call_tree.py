# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``ait.inspect(..., call_tree=True)`` on the GPU: the recorded tree, its CUDA times and the unchanged legacy output."""

import pytest
import torch

import aitune.torch as ait
from aitune.torch.call_tree.nodes import TensorLeaf
from tests.functional_ci.models import seeded
from tests.toy_models.call_tree_models import HIDDEN, VlaToy, vla_inputs

ITERATIONS = 4  # any count times every depth: the timing pass replays each sample once per call-path depth


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
    assert all(node.total_time_ms is not None and node.total_time_ms > 0 for node in tree.nodes.values())
    [dit] = tree.nodes_of(model.dit)
    assert dit.total_bytes_in > 0 and dit.total_bytes_out > 0


@pytest.mark.matrix("inspecting.call-tree/aot")
def test_call_tree_leaves_the_selected_modules_unchanged(device: torch.device):
    """``get_modules()`` selects the same modules with the same counts with and without ``call_tree=True``."""
    model, fn, dataset = _vla(device)

    without = _inspect(model, fn, dataset, call_tree=False)
    with_tree = _inspect(model, fn, dataset, call_tree=True)

    def legacy(info):
        return [(m.object_path, m.execution_count, m.forward_called) for m in info.get_modules()]

    assert legacy(with_tree) == legacy(without)


@pytest.mark.matrix("inspecting.call-tree-signatures/aot")
def test_call_tree_records_the_signatures_at_the_llm_to_dit_cut(device: torch.device):
    """The LLM's output and the DiT's inputs have one exact signature over all calls, and ``to_dict`` exports them."""
    model, fn, dataset = _vla(device)

    tree = _inspect(model, fn, dataset, call_tree=True).call_tree

    assert tree is not None
    llm = tree.nodes[("root.forward", "root.llm.forward")]
    [llm_signature] = llm.signatures
    assert llm_signature.call_indices == list(range(ITERATIONS))
    assert llm_signature.outputs == {"output": TensorLeaf("output", "torch.Tensor", torch.float32, (2, HIDDEN))}
    dit_key = ("root.forward", "root.dit_forward")
    [dit_signature] = tree.nodes[dit_key].signatures
    assert dit_signature.call_indices == list(range(5 * ITERATIONS))
    assert dit_signature.inputs == {
        "x": TensorLeaf("x", "torch.Tensor", torch.float32, (2, HIDDEN)),
        "t": TensorLeaf("t", "torch.Tensor", torch.float32, (2, 1)),
    }
    assert tree.calls_per_iteration()[dit_key] == [5] * ITERATIONS

    exported = tree.to_dict()
    [dit] = [node for node in exported["nodes"] if node["call_path"] == "forward > dit_forward"]
    assert dit["calls_per_iteration"] == [5] * ITERATIONS
    assert dit["signatures"][0]["inputs"]["t"] == {"type": "torch.Tensor", "dtype": "torch.float32", "shape": [2, 1]}
    tied = {"weight": {"dtype": "torch.float32", "shape": [16, HIDDEN]}}
    assert exported["modules"]["lm_head"]["parameters"] == tied
    assert exported["modules"]["llm.embed"]["parameters"] == tied
