# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend lifecycle cases on a toy MLP."""

from collections.abc import Callable
from functools import partial

import pytest
import torch

from aitune.torch.backend import Backend, TorchEagerBackend

from .lifecycle import Tolerance, run_lifecycle
from .models import make_input, make_mlp

BACKENDS: dict[str, tuple[Callable[[], Backend], Tolerance]] = {
    "torch-eager": (TorchEagerBackend, Tolerance()),
}


@pytest.mark.parametrize("backend_id", BACKENDS)
def test_backend_lifecycle(backend_id: str, device: torch.device, tmp_path):
    """Tune, infer, save, and reload the toy MLP with one backend."""
    backend_factory, tolerance = BACKENDS[backend_id]
    run_lifecycle(
        backend_factory(),
        partial(make_mlp, device),
        make_input(device),
        tmp_path / "model.ait",
        tolerance,
    )
