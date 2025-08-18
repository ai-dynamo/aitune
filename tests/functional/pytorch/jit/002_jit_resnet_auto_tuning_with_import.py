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
"""Test JIT tuning with auto tuning."""
# /// script
# dependencies = ["timm"]
# scope = "always"
# allow_failure = false
# ///

from io import StringIO
from logging import INFO, basicConfig

import timm
import torch

from aitune.torch.jit.patched_module import PatchedModule


def test_jit_resnet():
    import aitune.torch.jit.enable  # noqa: F401

    resnet = timm.create_model("resnet18", pretrained=False).to("cuda")

    def batch():
        # we are calling two times with different batch sizes to recognize dynamic axes
        resnet(torch.randn(2, 3, 224, 224, device="cuda"))
        resnet(torch.randn(16, 3, 224, 224, device="cuda"))

    for _ in range(5):
        batch()

    # Capture the print_hierarchy output
    with StringIO() as test_sink:
        PatchedModule.print_hierarchy(sink=test_sink.write)
        hierarchy_output = test_sink.getvalue()

    # Assert the expected output
    assert "PatchedModule Hierarchy:" in hierarchy_output
    assert "├─ ResNet 📊11.7M level=0🪜 state=tuned🎯 (TensorRTBackend)" in hierarchy_output

    assert resnet(torch.randn(8, 3, 224, 224, device="cuda")).shape == (8, 1000)
    assert resnet(torch.randn(16, 3, 224, 224, device="cuda")).shape == (16, 1000)


if __name__ == "__main__":
    basicConfig(level=INFO, force=True)
    test_jit_resnet()
