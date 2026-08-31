# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test Diffusers integrations."""

import importlib.util
import sys
import types
from pathlib import Path

import torch

from aitune.torch.integrations import is_integration_distributed_module
from aitune.torch.utils.module import is_distributed_module
from tests.toy_backends import SleepBackend

_DIFFUSERS_INTEGRATION_PATH = Path(__file__).parents[4] / "aitune" / "torch" / "integrations" / "diffusers.py"


def _install_fake_diffusers(monkeypatch, context_parallel_config: type | None = None):
    diffusers = types.ModuleType("diffusers")
    if context_parallel_config is not None:
        diffusers.ContextParallelConfig = context_parallel_config

    monkeypatch.setitem(sys.modules, "diffusers", diffusers)


def _load_diffusers_integration():
    spec = importlib.util.spec_from_file_location(
        "_aitune_torch_integrations_diffusers_test",
        _DIFFUSERS_INTEGRATION_PATH,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_diffusers_integration_keeps_native_rmsnorm(monkeypatch):
    original_rms_norm = torch.nn.RMSNorm
    original_normalization_rms_norm = torch.nn.modules.normalization.RMSNorm

    _install_fake_diffusers(monkeypatch)

    _load_diffusers_integration()

    assert torch.nn.RMSNorm is original_rms_norm
    assert torch.nn.modules.normalization.RMSNorm is original_normalization_rms_norm


def test_diffusers_context_parallel_detector_is_registered(monkeypatch):
    class ContextParallelConfig:
        pass

    _install_fake_diffusers(monkeypatch, ContextParallelConfig)
    module = torch.nn.Linear(2, 2)
    module._parallel_config = ContextParallelConfig()

    integration = _load_diffusers_integration()

    assert integration._is_context_parallel_module(module)
    assert is_integration_distributed_module(module)

    wrapped_module = torch.nn.Linear(2, 2)
    wrapped_module._parallel_config = types.SimpleNamespace(context_parallel_config=ContextParallelConfig())
    assert is_integration_distributed_module(wrapped_module)


def test_diffusers_context_parallel_detector_keeps_unparallelized_modules_single_gpu(monkeypatch):
    class ContextParallelConfig:
        pass

    _install_fake_diffusers(monkeypatch, ContextParallelConfig)
    parallelized_module = torch.nn.Linear(2, 2)
    parallelized_module._parallel_config = ContextParallelConfig()
    unparallelized_module = torch.nn.Linear(2, 2)

    _load_diffusers_integration()

    assert is_distributed_module(parallelized_module)
    assert not is_distributed_module(unparallelized_module)
    SleepBackend()._assert_execution_mode(unparallelized_module)
