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

    config.min_samples = 2
    config.dry_run = False
    config.detect_graph_breaks = False
    config.batch_axis_required = False

    def pre_hook(module, input):  # noqa: A002
        # this actually inject data into the model
        return torch.randn(2, 3, 224, 224, device="cuda")

    def post_hook(module, input, output):  # noqa: A002
        # this extract max detected element
        return torch.argmax(output, dim=1)

    resnet.register_forward_pre_hook(pre_hook)
    resnet.register_forward_hook(post_hook)

    with torch.no_grad():
        for _ in range(2):
            resnet()  # notice: not argument - it will be added by pre hook

    # Capture the print_hierarchy output
    history = []
    PatchedModule.print_hierarchy(sink=lambda s: history.append(s))
    print("\n".join(history))

    # Assert the expected output
    assert PRINT_HIERARCHY_HEADER in history[0]
    assert re.match(r".*ResNet.*state=tuned.*TensorRTBackend", history[1])

    # by calling this we are checking if hooks are fired even though TRT backend is used
    assert resnet().shape == (2,)


if __name__ == "__main__":
    basicConfig(level=INFO, force=True)
    test_jit_resnet()
