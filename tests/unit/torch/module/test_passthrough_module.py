# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for passhrough module."""

import pytest

from aitune.torch.module.passthrough_module import PassthroughModule
from aitune.torch.utils.cuda_utils import is_available as is_cuda_available
from tests.toy_models import ToyTorchModel

devices = ["cpu", "cuda"] if is_cuda_available() else ["cpu"]


@pytest.mark.parametrize("device", devices)
def test_call(device):
    model = ToyTorchModel()
    x = model.inputs()[0]
    module = PassthroughModule(model, device=device)

    module(x)
