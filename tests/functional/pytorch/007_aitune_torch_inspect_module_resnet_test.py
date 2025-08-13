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

from aitune.torch import inspect


def test_inspect_resnet50():
    # given
    model = timm.create_model("resnet50", pretrained=False)
    model.to("cuda")
    model.eval()
    data = torch.randn((3, 224, 224), device="cuda")

    # when
    modules_info = inspect(model, data)

    # then - verify inspection
    modules_info.describe()

    assert len(modules_info.get_modules()) == 1

    module_info = modules_info.get_modules()[0]
    assert module_info.name == model.__class__.__name__
    assert module_info.module_type == timm.models.resnet.ResNet
    assert module_info.execution_count == 1
    assert module_info.total_execution_time > 0
    assert module_info.average_execution_time > 0


if __name__ == "__main__":
    test_inspect_resnet50()
