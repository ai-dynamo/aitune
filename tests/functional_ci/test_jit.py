# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JIT mode cases: modules created under the patcher are tuned on live traffic."""

from typing import cast

import pytest
import torch
from torch import nn

import aitune.torch as ait
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.config import config as jit_config
from aitune.torch.jit.inspect_module import InspectModule
from aitune.torch.jit.patched_module import ModuleState, PatchedModule
from aitune.torch.jit.patcher import prepare_for_jit_tuning
from aitune.torch.jit.tune import deferred
from aitune.torch.module.tuned_module import TunedModule
from tests.functional_ci.backends import BackendCase, with_backends
from tests.functional_ci.lifecycle import assert_matches_eager
from tests.functional_ci.models import make_input, make_mlp


class JitMLP(nn.Module):
    """Single JIT head around the toy MLP; `nn.Sequential` containers are not patched, so their children would be heads."""

    def __init__(self, mlp: nn.Module):
        """Wrap the toy MLP."""
        super().__init__()
        self.mlp = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the toy MLP."""
        return self.mlp(x)


@pytest.mark.parametrize("mode", [JITMode.TUNE_EAGER, JITMode.TUNE_DEFERRED], ids=lambda mode: mode.value)
@with_backends("jit")
def test_jit_tuning(case: BackendCase, mode: JITMode, device: torch.device):
    """Record two batch sizes, tune in the requested mode, then serve an unseen in-range batch size."""
    backend, tolerance = case.factory(), case.tolerance
    jit_config.mode = mode
    jit_config.device = device
    jit_config.min_samples = 2
    jit_config.strategy = ait.OneBackendStrategy(backend).enable_performance_validation(False)
    reference = make_mlp(device)
    with prepare_for_jit_tuning():
        model = JitMLP(make_mlp(device))

    with torch.no_grad():
        model(make_input(device, batch_size=1))
        model(make_input(device, batch_size=4))
        if mode == JITMode.TUNE_DEFERRED:
            assert PatchedModule.heads[0]._state == ModuleState.RECORDING
            deferred()
            model(make_input(device, batch_size=1))

        (head,) = (head for head in PatchedModule.heads if head._name.split(" ")[0] == "JitMLP")
        assert head._state == ModuleState.TUNED
        (selected,) = cast(TunedModule, head._wrapper).backends.values()
        assert type(selected) is type(backend)
        assert_matches_eager(model, reference, device, tolerance, batch_sizes=(3,), seeds=(7,))


def test_jit_inspection_report(device: torch.device, tmp_path):
    """Inspect mode records the model hierarchy without tuning and writes an HTML report."""
    jit_config.mode = JITMode.INSPECT
    with prepare_for_jit_tuning():
        model = JitMLP(make_mlp(device))

    with torch.no_grad():
        model(make_input(device, batch_size=1))
        model(make_input(device, batch_size=2))

    assert [head._name.split(" ")[0] for head in InspectModule.heads] == ["JitMLP"]
    report = tmp_path / "inspection.html"
    InspectModule.save_report(report, "toy-mlp")
    assert "toy-mlp" in report.read_text()
    assert not PatchedModule.heads
