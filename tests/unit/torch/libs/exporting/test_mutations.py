# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend-independent checks for Torch Export mutation preservation."""

import pytest
import torch
import torch.nn as nn

from tests.toy_models.stateful import MUTATION_OPERATIONS, update_tensor


class CopyBackModel(nn.Module):
    def __init__(self, operation="add_"):
        super().__init__()
        self.operation = operation
        self.register_buffer("counter", torch.tensor(0))

    def forward(self, value, cache, index):
        update_tensor(cache, value.sin(), index, self.operation)
        self.counter.add_(1)
        return cache, self.counter


@pytest.mark.parametrize("operation", MUTATION_OPERATIONS)
@pytest.mark.parametrize("reexport", [False, True], ids=["export", "public-reexport"])
def test_copy_back_serialization_round_trip(tmp_path, operation, reexport):
    model = CopyBackModel(operation)
    sample = (torch.ones(3), torch.ones(3), torch.arange(3).flip(0))
    dynamic_shapes = ({0: torch.export.Dim("batch", min=2, max=8)},) * 3
    original = torch.export.export(model, sample, dynamic_shapes=dynamic_shapes).run_decompositions()
    if reexport:
        original = torch.export.export(
            original.module(), sample, dynamic_shapes=dynamic_shapes, strict=False
        ).run_decompositions()
    original.validate()
    assert len(original.graph_signature.user_inputs_to_mutate) == 1
    assert len(original.graph_signature.buffers_to_mutate) == 1
    path = tmp_path / "stateful.pt2"
    torch.export.save(original, path)
    loaded = torch.export.load(path).module()
    expected_model = CopyBackModel(operation)
    actual_cache, expected_cache = torch.ones(5), torch.ones(5)
    for scale in (2.75, 0.5, -1.0):
        value = torch.arange(5).float() + scale
        index = torch.arange(5).roll(1)
        actual = loaded(value, actual_cache, index)
        expected = expected_model(value, expected_cache, index)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_cache, expected_cache)
        assert actual[0] is actual_cache
        assert actual[1] is loaded.counter


def test_copy_backs_are_inserted_by_unlift_without_aitune_wrappers():
    model = CopyBackModel()
    exported = torch.export.export(model, (torch.ones(3), torch.ones(3), torch.arange(3))).run_decompositions()
    assert not any(node.target == torch.ops.aten.copy_.default for node in exported.graph.nodes)
    writes = [node for node in exported.module().graph.nodes if node.target == torch.ops.aten.copy_.default]
    assert len(writes) == 2
