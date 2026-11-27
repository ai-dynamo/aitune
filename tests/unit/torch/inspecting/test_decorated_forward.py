# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
import torch

import aitune.torch as ait
from tests.toy_models.decorated_forward_models import ToyDecoratedSpeechModel


@pytest.mark.parametrize("inspect_first", [False, True], ids=["fresh", "inspected"])
def test_decorated_forward_model_can_be_copied(inspect_first):
    model = ToyDecoratedSpeechModel().eval()
    sample = torch.ones(2, 8)
    if inspect_first:
        ait.inspect(model, [sample], number_of_iterations=1, warmup_iterations=1)

    copied = deepcopy(model)
    with torch.no_grad():
        original_output = model(sample)
        torch.testing.assert_close(copied(sample), original_output)
        copied.encoder.weight.zero_()
        # A copied forward must read the copy's weights even after inspection.
        torch.testing.assert_close(copied(sample), torch.zeros_like(original_output))
        torch.testing.assert_close(model(sample), original_output)


@pytest.mark.parametrize("inspect_first", [False, True], ids=["fresh", "inspected"])
def test_dynamo_compiles_decorated_forward_without_aitune_backend(inspect_first):
    model = ToyDecoratedSpeechModel().eval()
    sample = torch.ones(2, 8)
    if inspect_first:
        ait.inspect(model, [sample], number_of_iterations=1, warmup_iterations=1)

    torch._dynamo.reset()
    try:
        with torch.no_grad():
            expected = model(sample)
            # The eager compiler backend still exercises Dynamo's guards without requiring GPU kernels.
            compiled = torch.compile(model, backend="eager", fullgraph=False, dynamic=False)
            torch.testing.assert_close(compiled(sample), expected)
    finally:
        torch._dynamo.reset()
