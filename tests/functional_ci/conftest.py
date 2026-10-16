# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the toy-model functional suite.

The root `conftest.py` already isolates the AITune cache, module registry, and JIT config per test.
"""

import pytest
import torch

from aitune.torch.config import config
from aitune.torch.module_registry import MODULE_REGISTRY


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Return the first GPU; the suite fails instead of skipping without CUDA."""
    if not torch.cuda.is_available():
        pytest.fail("tests/functional_ci requires CUDA", pytrace=False)
    return torch.device("cuda:0")


@pytest.fixture(autouse=True)
def isolated_tuning_report(monkeypatch, tmp_path):
    """Keep tuning reports out of the user cache and release backend resources after each case."""
    monkeypatch.setattr(config, "_tuning_data_output_path", tmp_path / "tuning-report.json")
    yield
    for module in MODULE_REGISTRY.modules.values():
        module.deactivate()
