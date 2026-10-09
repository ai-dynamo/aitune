# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for required functional CI cases."""

from collections.abc import Iterator

import pytest
import torch

from aitune.torch.config import config
from aitune.torch.module_registry import MODULE_REGISTRY


@pytest.fixture(scope="session")
def functional_device() -> torch.device:
    """Require the declared GPU runner and return its explicit primary device."""
    if not torch.cuda.is_available():
        pytest.fail("Functional CI requires CUDA, but torch.cuda.is_available() is false", pytrace=False)
    return torch.device("cuda:0")


@pytest.fixture(autouse=True)
def functional_state(tmp_path, monkeypatch) -> Iterator[None]:
    """Keep tuning reports out of the user cache and release backends after each case."""
    monkeypatch.setattr(config, "_tuning_data_output_path", tmp_path / "tuning-report.json")
    yield
    for module in MODULE_REGISTRY.modules.values():
        module.deactivate()
