# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest fixtures both for unit and integration tests and docttest for production code."""

import logging
import os

import pytest
import torch

from aitune.torch.jit.config import config as jit_config
from aitune.torch.jit.patcher import jit_reset
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.tune_data.reporting import _active_graph, _active_module, _active_report, _run_start_ts
from aitune.torch.utils.cuda_utils import is_available as is_cuda_available
from aitune.utils.logging import setup_logging


@pytest.fixture(autouse=True)
def module_registry_cleanup():
    """Cleans up the module registry after each test."""
    try:
        yield
    finally:
        MODULE_REGISTRY.clear()


@pytest.fixture(autouse=True)
def aitune_cache_dir(mocker, tmp_path):
    """Sets cache dir to temporary directory."""
    import sys

    # aitune.torch.config resolves to the config object due to import in aitune.torch.__init__
    # so we fetch the module directly from sys.modules
    config_module = sys.modules["aitune.torch.config"]

    cache_dir = tmp_path / "aitune_cache"
    mocker.patch.object(config_module, "_AITUNE_CACHE_DIR", cache_dir)
    return cache_dir


@pytest.fixture
def torch_device():
    """Returns the device specified in AITUNE_TESTS_USE_DEVICE environment variable if set,
    otherwise returns CUDA if available, or CPU as fallback.
    """  # noqa: D205
    user_device = os.environ.get("AITUNE_TESTS_USE_DEVICE")

    if user_device is not None:
        assert is_cuda_available(), "CUDA is not available, but AITUNE_TESTS_USE_DEVICE is set to CUDA"
        return torch.device(user_device)

    return torch.device("cuda" if is_cuda_available() else "cpu")


@pytest.fixture(autouse=True)
def jit_cleanup():
    """Reset patcher state and jit_config before and after each test."""
    jit_reset()
    _reset_tuning_report_context()
    jit_config.reset_to_defaults()
    try:
        yield
    finally:
        jit_reset()
        _reset_tuning_report_context()
        jit_config.reset_to_defaults()


@pytest.fixture(autouse=True)
def aitune_logging_setup():
    """Setup logging for aitune."""
    setup_logging(level=logging.DEBUG if os.environ.get("AITUNE_TESTS_LOG_LEVEL") == "DEBUG" else logging.INFO)
    yield
    logging.shutdown()


def _reset_tuning_report_context():
    """Clear in-progress tuning report state between tests."""
    _active_report.set(None)
    _active_module.set(None)
    _active_graph.set(None)
    _run_start_ts.set(None)
