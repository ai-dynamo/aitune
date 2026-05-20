# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test Diffusers integrations."""

import importlib.util
import sys
import types
from pathlib import Path

import torch

_DIFFUSERS_INTEGRATION_PATH = Path(__file__).parents[4] / "aitune" / "torch" / "integrations" / "diffusers.py"


def _install_fake_diffusers(monkeypatch, version: str, rms_norm: type):
    diffusers = types.ModuleType("diffusers")
    diffusers.__version__ = version
    diffusers_models = types.ModuleType("diffusers.models")
    diffusers_normalization = types.ModuleType("diffusers.models.normalization")
    diffusers_normalization.RMSNorm = rms_norm

    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.models", diffusers_models)
    monkeypatch.setitem(sys.modules, "diffusers.models.normalization", diffusers_normalization)


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


def test_diffusers_rmsnorm_patch_is_applied_for_diffusers_0_35(monkeypatch):
    class DiffusersRMSNorm(torch.nn.Module):
        pass

    monkeypatch.setattr(torch.nn, "RMSNorm", torch.nn.RMSNorm)
    monkeypatch.setattr(torch.nn.modules.normalization, "RMSNorm", torch.nn.modules.normalization.RMSNorm)
    _install_fake_diffusers(monkeypatch, "0.35.0", DiffusersRMSNorm)

    _load_diffusers_integration()

    assert torch.nn.RMSNorm is DiffusersRMSNorm
    assert torch.nn.modules.normalization.RMSNorm is DiffusersRMSNorm


def test_diffusers_rmsnorm_patch_is_skipped_before_diffusers_0_35(monkeypatch):
    original_rms_norm = torch.nn.RMSNorm
    original_normalization_rms_norm = torch.nn.modules.normalization.RMSNorm

    class DiffusersRMSNorm(torch.nn.Module):
        pass

    monkeypatch.setattr(torch.nn, "RMSNorm", original_rms_norm)
    monkeypatch.setattr(torch.nn.modules.normalization, "RMSNorm", original_normalization_rms_norm)
    _install_fake_diffusers(monkeypatch, "0.34.2", DiffusersRMSNorm)

    _load_diffusers_integration()

    assert torch.nn.RMSNorm is original_rms_norm
    assert torch.nn.modules.normalization.RMSNorm is original_normalization_rms_norm
