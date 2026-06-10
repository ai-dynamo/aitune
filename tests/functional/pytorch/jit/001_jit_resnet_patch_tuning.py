# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test JIT tuning with patch decorator on resnet."""
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
from aitune.torch.jit.patcher import patch_for_jit_tuning


@patch_for_jit_tuning
def create_resnet():
    """Create a ResNet18 model.

    The decorator will make this model tunable.
    """
    return timm.create_model("resnet18", pretrained=False).to("cuda")


def test_jit_resnet():
    resnet = create_resnet()

    config.min_samples = 4  # just to compare before and after tuning
    config.dry_run = False
    config.detect_graph_breaks = False

    def batch():
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
    assert re.match(r".*ResNet.*state=tuned.*", history[1])

    assert resnet(torch.randn(8, 3, 224, 224, device="cuda")).shape == (8, 1000)
    assert resnet(torch.randn(16, 3, 224, 224, device="cuda")).shape == (16, 1000)


if __name__ == "__main__":
    basicConfig(level=INFO, force=True)
    test_jit_resnet()
