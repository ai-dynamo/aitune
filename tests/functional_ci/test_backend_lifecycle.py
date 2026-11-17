# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend lifecycle and exported-artifact cases on a toy MLP."""

from functools import partial
from typing import cast

import pytest
import torch

import aitune.torch as ait
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.config import config as jit_config
from aitune.torch.jit.patched_module import ModuleState, PatchedModule
from aitune.torch.jit.patcher import prepare_for_jit_tuning
from aitune.torch.module.tuned_module import TunedModule
from tests.functional_ci.artifacts import run_exported_artifact
from tests.functional_ci.backends import BackendCase, with_backends
from tests.functional_ci.lifecycle import assert_matches_eager, pinned_strategy, run_lifecycle
from tests.functional_ci.models import make_input, make_mlp
from tests.functional_ci.test_jit import JitMLP


@pytest.mark.matrix(
    "aot_tuning.models.tensor-inputs",
    "aot_tuning.models.static-shapes",
    "aot_tuning.models.single-graph",
    "modes.aot-tune",
    "modes.aot-save-load",
    "topology.{backend}/single-gpu",
    "topology.inspect/single-gpu",
    "strategies.one-backend/aot-tune",
)
@with_backends("base")
def test_backend_lifecycle(case: BackendCase, device: torch.device, tmp_path):
    """Tune, infer, save, and reload the toy MLP."""
    run_lifecycle(case.factory(), partial(make_mlp, device), make_input(device), tmp_path / "model.ait", case.tolerance)


@pytest.mark.matrix("jit_tuning.models.static-shapes")
@with_backends("base")
def test_backend_lifecycle_jit(case: BackendCase, device: torch.device):
    """Tune from a single batch size and serve the same static shape."""
    backend = case.factory()
    jit_config.mode = JITMode.TUNE_EAGER
    jit_config.device = device
    jit_config.min_samples = 1
    jit_config.batch_axis_required = False
    jit_config.strategy = pinned_strategy(backend)
    reference = make_mlp(device)
    with prepare_for_jit_tuning():
        model = JitMLP(make_mlp(device))

    with torch.no_grad():
        model(make_input(device, batch_size=2))
    (head,) = PatchedModule.heads
    assert head._state == ModuleState.TUNED
    (selected,) = cast(TunedModule, head._wrapper).backends.values()
    assert type(selected) is type(backend)
    assert_matches_eager(model, reference, device, case.tolerance, batch_sizes=(2,))


@pytest.mark.matrix("modes.aot-export")
@with_backends("export")
def test_exported_artifact(case: BackendCase, device: torch.device, tmp_path):
    """The tuned module exports a deployable artifact that runs natively, without AITune, and matches eager."""
    tuned = ait.Module(make_mlp(device), "mlp", strategy=pinned_strategy(case.factory()))
    ait.tune(tuned, list(make_input(device).unbind(0)), batch_sizes=[2], device=device)
    artifact = tuned.artifact()
    assert artifact.model.format == case.artifact_format

    def run_native(value: torch.Tensor) -> torch.Tensor:
        (output,) = run_exported_artifact(artifact, tmp_path / "exported", [value])
        return output

    assert_matches_eager(run_native, make_mlp(device), device, case.tolerance)
