# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test JIT tuning with TorchAOBackend and an unobserved child module."""
# /// script
# scope = "always"
# allow_failure = false
# ///

import re
from logging import INFO, basicConfig

import torch
from _tuning_data_artifacts import collect_tuning_data

from aitune.torch.backend.torchao_backend import TorchAOBackend, TorchAOBackendConfig
from aitune.torch.jit.config import JITMode, config
from aitune.torch.jit.patched_module import PRINT_HIERARCHY_HEADER, PatchedModule
from aitune.torch.jit.patcher import patch_for_jit_tuning
from aitune.torch.jit.tune import deferred as tune_deferred
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy


class ModelWithUnusedChild(torch.nn.Module):
    """Small model with a registered child that is not called by forward."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
        self.unused = torch.nn.Identity()

    def forward(self, x):
        return self.linear(x)


@patch_for_jit_tuning
def create_model():
    """Create a model with an unobserved child module."""
    return ModelWithUnusedChild().to("cuda", dtype=torch.float16).eval()


@collect_tuning_data(__file__)
def test_jit_torchao_unused_child():
    model = create_model()

    strategy = OneBackendStrategy(
        backend=TorchAOBackend(config=TorchAOBackendConfig(quantization="int8wo")),
    )
    strategy.enable_find_max_batch_size(False)
    strategy.enable_performance_validation(False)

    config.strategy = strategy
    config.mode = JITMode.TUNE_DEFERRED
    config.min_samples = 1
    config.dry_run = False
    config.detect_graph_breaks = False
    config.batch_axis_required = False

    with torch.no_grad():
        model(torch.randn(2, 16, device="cuda", dtype=torch.float16))

    tune_deferred()

    history = []
    PatchedModule.print_hierarchy(sink=lambda s: history.append(s))
    print("\n".join(history))

    assert PRINT_HIERARCHY_HEADER in history[0]
    assert re.match(r".*ModelWithUnusedChild.*state=tuned.*TorchAOBackend", history[1])

    with torch.no_grad():
        assert model(torch.randn(2, 16, device="cuda", dtype=torch.float16)).shape == (2, 16)


if __name__ == "__main__":
    basicConfig(level=INFO, force=True)
    test_jit_torchao_unused_child()
