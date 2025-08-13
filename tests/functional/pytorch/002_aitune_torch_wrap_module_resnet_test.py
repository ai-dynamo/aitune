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
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = ["timm"]
#
# # Optional, default "always", determines how often test is generated, always, nightly, weekly, monthly
# scope = "always"
# ///


import timm
import torch

from aitune.torch.backend.torch_inductor_backend import TorchInductorBackend
from aitune.torch.module.wrapper_module import Module
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy


def test_resnet50():
    # given
    device = torch.device("cuda")

    model = timm.create_model("resnet50", pretrained=False)
    model.to(device)
    model.eval()
    data = torch.randn((2, 3, 224, 224), device=device)

    with torch.inference_mode():
        out = model(data)
    expected_probs = torch.nn.functional.softmax(out[0], dim=0)

    # when
    module = Module(model, "functional-resnet50")

    # then - verify recording
    module(data)
    assert len(module.graph_specs) == 1

    # then - verify tuning
    strategy = OneBackendStrategy(TorchInductorBackend())
    module.tune(device=device, strategy=strategy, dry_run=False)
    out = module(data)
    actual_probs = torch.nn.functional.softmax(out[0], dim=0)
    torch.testing.assert_close(actual_probs, expected_probs, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    test_resnet50()
