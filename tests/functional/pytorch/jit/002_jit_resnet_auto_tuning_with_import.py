# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test JIT tuning with auto tuning."""
# /// script
# dependencies = ["timm"]
# scope = "always"
# allow_failure = false
# ///

import re
from logging import INFO, basicConfig

import timm
import torch

from aitune.torch.jit.config import config
from aitune.torch.jit.patched_module import PRINT_HIERARCHY_HEADER, PatchedModule


def test_jit_resnet():
    import aitune.torch.jit.enable  # noqa: F401

    config.min_samples = 2

    resnet = timm.create_model("resnet18", pretrained=False).to("cuda")

    def batch():
        # we are calling two times with different batch sizes to recognize dynamic axes
        resnet(torch.randn(2, 3, 224, 224, device="cuda"))
        resnet(torch.randn(16, 3, 224, 224, device="cuda"))

    for _ in range(5):
        batch()

    # Capture the print_hierarchy output
    history = []
    PatchedModule.print_hierarchy(sink=lambda s: history.append(s))
    print("\n".join(history))

    # Assert the expected output
    assert PRINT_HIERARCHY_HEADER in history[0]
    assert re.match(r".*ResNet.*state=tuned.*TensorRTBackend", history[1])

    assert resnet(torch.randn(8, 3, 224, 224, device="cuda")).shape == (8, 1000)
    assert resnet(torch.randn(16, 3, 224, 224, device="cuda")).shape == (16, 1000)


if __name__ == "__main__":
    basicConfig(level=INFO, force=True)
    test_jit_resnet()
