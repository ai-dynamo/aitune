# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test the patcher functions."""

from unittest.mock import Mock

import torch

from aitune.torch.jit.config import config as jit_config
from aitune.torch.jit.patcher import Patcher, jit_reset, patch_for_jit_tuning, prepare_for_jit_tuning


def _module_with_module_name(module_name: str) -> torch.nn.Module:
    module_cls = type("SyntheticModule", (torch.nn.Module,), {"__module__": module_name})
    return module_cls()


def test_jit_reset():
    """Test jit_reset function."""
    Patcher.patch_torch()
    torch.nn.Linear(10, 5)
    assert len(Patcher._patched_modules) == 1
    jit_reset()
    assert len(Patcher._patched_modules) == 0
    assert len(Patcher._intercepted_classes) == 0
    torch.nn.Linear(10, 5)
    assert len(Patcher._patched_modules) == 0
    assert len(Patcher._intercepted_classes) == 0


def test_prepare_for_tuning():
    """Test prepare_for_tuning context manager."""
    pre_module = torch.nn.Linear(10, 5)  # noqa: F841
    with prepare_for_jit_tuning():
        module = torch.nn.Linear(10, 5)
    after_module = torch.nn.Linear(10, 5)  # noqa: F841
    assert len(Patcher._patched_modules) == 1
    patched_module = Patcher._patched_modules[0]
    assert patched_module.__wrapped__ == module


def test_patch_decorator():
    """Test prepare_for_tuning context manager."""

    @patch_for_jit_tuning
    def create_module():
        module = torch.nn.Linear(10, 5)
        return module

    pre_module = torch.nn.Linear(10, 5)  # noqa: F841
    module = create_module()
    after_module = torch.nn.Linear(10, 5)  # noqa: F841
    assert len(Patcher._patched_modules) == 1
    patched_module = Patcher._patched_modules[0]
    assert patched_module.__wrapped__ == module


def test_is_allowed_to_tune():
    """Test is_allowed_to_tune function."""
    assert Patcher._is_allowed_to_tune(torch.nn.Linear(10, 5))
    assert not Patcher._is_allowed_to_tune(Mock(spec=torch._dynamo.eval_frame.OptimizedModule))


def test_is_allowed_to_tune_excludes_submodule_classes():
    """Exclusions are by prefix — classes nested under an excluded module are also rejected.

    Regression for the wrapt-in-gm save crash: torch_tensorrt builds
    ``torch_tensorrt.dynamo.runtime._TorchTensorRTModule.TorchTensorRTModule``
    instances during compile and inserts them into the compiled gm. Wrapping
    their ``forward`` with wrapt makes the subsequent ``torch_tensorrt.save``
    deepcopy raise. The exclude check must therefore match by module prefix.
    """
    module = _module_with_module_name("torch_tensorrt.dynamo.runtime._TorchTensorRTModule")

    assert not Patcher._is_allowed_to_tune(module)


def test_is_allowed_to_tune_allows_unrelated_submodule():
    """A class whose module prefix is *not* in the exclude list stays tunable."""
    module = _module_with_module_name("some.other.library")

    assert Patcher._is_allowed_to_tune(module)


def test_is_allowed_to_tune_extends_patch_exclude_with_user_module_prefix_entries(monkeypatch):
    """User-supplied patch_exclude module prefixes add to the built-in defaults."""
    monkeypatch.setattr(jit_config, "patch_exclude", ("my_blocked_pkg",))
    module = _module_with_module_name("my_blocked_pkg.submodule")

    assert not Patcher._is_allowed_to_tune(module)


def test_is_allowed_to_tune_extends_patch_exclude_with_user_exact_module_entries(monkeypatch):
    """User-supplied patch_exclude module prefixes can match the exact module name."""
    monkeypatch.setattr(jit_config, "patch_exclude", ("foo.bar",))
    module = _module_with_module_name("foo.bar")

    assert not Patcher._is_allowed_to_tune(module)


def test_is_allowed_to_tune_extends_patch_exclude_with_user_class_entries(monkeypatch):
    """User-supplied patch_exclude module class FQNs add to the built-in defaults."""
    monkeypatch.setattr(jit_config, "patch_exclude", ("torch.nn.modules.linear.Linear",))

    assert not Patcher._is_allowed_to_tune(torch.nn.Linear(10, 5))
    # Built-in defaults still apply alongside user-supplied entries.
    assert not Patcher._is_allowed_to_tune(torch.nn.ModuleList())


def test_is_allowed_to_tune_rejects_bare_class_name_entries(monkeypatch):
    """Bare class-name patch_exclude entries are not treated as class FQNs."""
    monkeypatch.setattr(jit_config, "patch_exclude", ("Linear",))

    assert Patcher._is_allowed_to_tune(torch.nn.Linear(10, 5))


def test_is_allowed_to_tune_rejects_torch_nn_alias_entries(monkeypatch):
    """torch.nn alias patch_exclude entries are not treated as class FQNs."""
    monkeypatch.setattr(jit_config, "patch_exclude", ("torch.nn.Linear",))

    assert Patcher._is_allowed_to_tune(torch.nn.Linear(10, 5))


def test_is_allowed_to_tune_user_cannot_remove_default_exclusions(monkeypatch):
    """Clearing the user-config tuple does not disable the built-in defaults."""
    monkeypatch.setattr(jit_config, "patch_exclude", ())

    assert not Patcher._is_allowed_to_tune(torch.nn.ModuleList())


def test_intercepted_classes():
    """Test intercepted_classes function."""

    @patch_for_jit_tuning
    def create_module():
        module = torch.nn.Linear(10, 5)
        return module

    _ = create_module()
    assert Patcher.intercepted_classes() == ["torch.nn.modules.linear.Linear"]
