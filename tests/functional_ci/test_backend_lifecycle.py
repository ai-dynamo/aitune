# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend lifecycle and exported-artifact cases on a toy MLP."""

from functools import partial

import pytest
import torch

import aitune.torch as ait
from tests.functional_ci.artifacts import run_exported_artifact
from tests.functional_ci.backends import BackendCase, with_backends
from tests.functional_ci.lifecycle import assert_matches_eager, pinned_strategy, run_lifecycle
from tests.functional_ci.models import make_input, make_mlp


@pytest.mark.matrix(
    "models.tensor-inputs",
    "models.static-shapes",
    "models.single-graph",
    "modes.aot-tune",
    "modes.aot-save-load",
    "topology.{backend}/single-gpu",
    "strategies.one-backend/aot-tune",
)
@with_backends("save_load")
def test_backend_lifecycle(case: BackendCase, device: torch.device, tmp_path):
    """Tune, infer, save, and reload the toy MLP."""
    run_lifecycle(case.factory(), partial(make_mlp, device), make_input(device), tmp_path / "model.ait", case.tolerance)


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
