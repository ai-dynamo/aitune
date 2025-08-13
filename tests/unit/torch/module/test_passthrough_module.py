# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific
"""Test for passhrough module."""

import pytest

from aitune.torch.module.passthrough_module import PassthroughModule
from aitune.torch.utils.cuda import is_available as is_cuda_available
from tests.toy_models import ToyTorchModel

devices = ["cpu", "cuda"] if is_cuda_available() else ["cpu"]


@pytest.mark.parametrize("device", devices)
def test_call(device):
    model = ToyTorchModel()
    x = model.inputs()[0]
    module = PassthroughModule(model, device=device)

    module(x)
