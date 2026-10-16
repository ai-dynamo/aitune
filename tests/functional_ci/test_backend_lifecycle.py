# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend lifecycle cases on a toy MLP."""

from functools import partial

import torch

from tests.functional_ci.artifacts import run_exported_artifact
from tests.functional_ci.backends import BackendCase, with_backends
from tests.functional_ci.lifecycle import run_lifecycle
from tests.functional_ci.models import make_input, make_mlp


@with_backends("save_load")
def test_backend_lifecycle(case: BackendCase, device: torch.device, tmp_path):
    """Tune, infer, save, and reload the toy MLP; run the exported artifact natively where deployable."""
    value = make_input(device)
    result = run_lifecycle(case.factory(), partial(make_mlp, device), value, tmp_path / "model.ait", case.tolerance)
    if case.artifact_format is None:
        return

    artifact = result.restored.artifact()
    assert artifact.model.format == case.artifact_format
    (output,) = run_exported_artifact(artifact, tmp_path / "exported", [value])
    with torch.no_grad():
        expected = make_mlp(device)(value)
    torch.testing.assert_close(output, expected, rtol=case.tolerance.rtol, atol=case.tolerance.atol)
