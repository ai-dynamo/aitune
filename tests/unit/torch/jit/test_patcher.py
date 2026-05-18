# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test the patcher functions."""

from unittest.mock import Mock

import torch

from aitune.torch.jit.config import config as jit_config
from aitune.torch.jit.patcher import Patcher, jit_reset, patch_for_jit_tuning, prepare_for_jit_tuning


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


def test_is_allowed_to_tune_excludes_subpackage_classes(mocker):
    """Exclusions are by prefix — classes nested under an excluded package are also rejected.

    Regression for the wrapt-in-gm save crash: torch_tensorrt builds
    ``torch_tensorrt.dynamo.runtime._TorchTensorRTModule.TorchTensorRTModule``
    instances during compile and inserts them into the compiled gm. Wrapping
    their ``forward`` with wrapt makes the subsequent ``torch_tensorrt.save``
    deepcopy raise. The exclude check must therefore match by package prefix.
    """
    module = torch.nn.Module()
    fake_module_info = Mock(__package__="torch_tensorrt.dynamo.runtime._TorchTensorRTModule")
    mocker.patch("aitune.torch.jit.patcher.inspect.getmodule", return_value=fake_module_info)

    assert not Patcher._is_allowed_to_tune(module)


def test_is_allowed_to_tune_allows_unrelated_subpackage(mocker):
    """A class whose package prefix is *not* in the exclude list stays tunable."""
    module = torch.nn.Module()
    fake_module_info = Mock(__package__="some.other.library")
    mocker.patch("aitune.torch.jit.patcher.inspect.getmodule", return_value=fake_module_info)

    assert Patcher._is_allowed_to_tune(module)


def test_is_allowed_to_tune_extends_extra_patch_exclude_packages_with_user_entries(mocker):
    """User-supplied extra_patch_exclude_packages add to the built-in defaults."""
    jit_config.extra_patch_exclude_packages = ("my_blocked_pkg",)
    module = torch.nn.Module()
    fake_module_info = Mock(__package__="my_blocked_pkg.submodule")
    mocker.patch("aitune.torch.jit.patcher.inspect.getmodule", return_value=fake_module_info)

    assert not Patcher._is_allowed_to_tune(module)


def test_is_allowed_to_tune_extends_extra_patch_exclude_modules_with_user_entries():
    """User-supplied extra_patch_exclude_modules add to the built-in defaults."""
    jit_config.extra_patch_exclude_modules = (torch.nn.Linear,)

    assert not Patcher._is_allowed_to_tune(torch.nn.Linear(10, 5))
    # Built-in defaults still apply alongside user-supplied entries.
    assert not Patcher._is_allowed_to_tune(torch.nn.ModuleList())


def test_is_allowed_to_tune_user_cannot_remove_default_exclusions():
    """Clearing the user-config tuple does not disable the built-in defaults."""
    jit_config.extra_patch_exclude_modules = ()

    assert not Patcher._is_allowed_to_tune(torch.nn.ModuleList())


def test_intercepted_classes():
    """Test intercepted_classes function."""

    @patch_for_jit_tuning
    def create_module():
        module = torch.nn.Linear(10, 5)
        return module

    _ = create_module()
    assert Patcher.intercepted_classes() == ["torch.nn.modules.linear.Linear"]
