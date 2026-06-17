# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test JIT tuning on resnet pinned to TorchTensorRTAotBackend."""
# /// script
# dependencies = ["timm"]
# scope = "always"
# allow_failure = false
# ///

import re
from logging import INFO, basicConfig

import timm
import torch
from _tuning_data_artifacts import collect_tuning_data

from aitune.torch.backend.torch_tensorrt_aot_backend import TorchTensorRTAotBackend
from aitune.torch.jit.config import config
from aitune.torch.jit.patched_module import PRINT_HIERARCHY_HEADER, PatchedModule
from aitune.torch.jit.patcher import patch_for_jit_tuning
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy


@patch_for_jit_tuning
def create_resnet():
    """Create a ResNet18 model.

    The decorator will make this model tunable.
    """
    return timm.create_model("resnet18", pretrained=False).to("cuda")


@collect_tuning_data(__file__)
def test_jit_resnet():
    resnet = create_resnet()

    config.min_samples = 4
    config.dry_run = False
    config.detect_graph_breaks = False
    # The regression we are guarding (wrapt-decorated forward on
    # TorchTensorRTAotBackend's ``_run_on_acc_0`` crashing ``torch_tensorrt.save``'s
    # deepcopy) is independent of dynamic shapes. Run with a single static batch
    # size to keep the engine activation memory small enough for memory-constrained
    # CI runners — the dynamic-shape engine builder otherwise over-allocates ~20 GB.
    config.batch_axis_required = False
    config.strategy = OneBackendStrategy(backend=TorchTensorRTAotBackend())

    def batch():
        resnet(torch.randn(2, 3, 224, 224, device="cuda"))

    with torch.no_grad():
        for _ in range(5):
            batch()

    # Capture the print_hierarchy output
    history = []
    PatchedModule.print_hierarchy(sink=lambda s: history.append(s))
    print("\n".join(history))

    # Assert the expected output
    assert PRINT_HIERARCHY_HEADER in history[0]
    assert re.match(r".*ResNet.*state=tuned.*TorchTensorRTAotBackend", history[1])

    assert resnet(torch.randn(2, 3, 224, 224, device="cuda")).shape == (2, 1000)


if __name__ == "__main__":
    basicConfig(level=INFO, force=True)
    test_jit_resnet()
