# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate compilation of fresh and inspected models with decorated child forwards."""

import pytest
import torch

from aitune.torch.inspecting import inspect
from tests.toy_models.decorated_forward_models import ToyDecoratedSpeechModel


@pytest.mark.parametrize("inspect_before_compile", [False, True], ids=["fresh-model", "inspected-model"])
def test_compile_decorated_child_forward_after_graph_break(inspect_before_compile):
    # Job 573302209 compiles successfully after inspection, then fails when loading
    # into a fresh model. Inspection restores bound forwards as instance attributes;
    # a fresh model instead resolves the wrapt descriptor from the class dictionary.
    # Use Dynamo's eager backend to isolate guard generation from GPU code generation.
    torch._dynamo.reset()
    try:
        model = ToyDecoratedSpeechModel().eval()
        inputs = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        with torch.no_grad():
            expected = model(inputs)
        if inspect_before_compile:
            inspect(model, inputs, number_of_iterations=1, warmup_iterations=1)

        compiled = torch.compile(model, backend="eager", fullgraph=False, dynamic=False)
        with torch.no_grad():
            actual = compiled(inputs)
        torch.testing.assert_close(actual, expected)
    finally:
        torch._dynamo.reset()
