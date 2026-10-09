# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test the patcher functions."""

from importlib import import_module
from unittest.mock import Mock

import pytest
import torch

from aitune.torch.jit.config import JITMode
from aitune.torch.jit.config import config as jit_config
from aitune.torch.jit.inspect_module import InspectModule
from aitune.torch.jit.patched_module import PatchedModule
from aitune.torch.jit.patcher import Patcher, jit_reset, patch_for_jit_tuning, prepare_for_jit_tuning
from aitune.torch.tune_data.reporting import has_active_report


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


def test_eager_jit_tunes_when_ready(mocker):
    jit_config.mode = JITMode.TUNE_EAGER
    tune = mocker.patch.object(PatchedModule, "tune")

    with prepare_for_jit_tuning():
        module = torch.nn.Linear(2, 2)

    with torch.no_grad():
        module(torch.ones(1, 2))

    tune.assert_called_once_with()


def test_activation_allows_configuring_deferred_mode_before_first_eligible_module(mocker):
    jit_config.mode = JITMode.TUNE_EAGER
    Patcher.patch_torch()
    assert Patcher._exit_handler is not None
    assert Patcher._session_mode is None
    assert not has_active_report()

    jit_config.mode = JITMode.TUNE_DEFERRED
    jit_config.device = torch.device("cpu")
    jit_config.batch_axis_required = False
    tune = mocker.patch.object(PatchedModule, "tune")
    module = torch.nn.Linear(2, 2)

    assert Patcher._session_mode is JITMode.TUNE_DEFERRED
    assert Patcher._session_report is not None
    assert has_active_report()
    module(torch.ones(1, 2))
    tune.assert_not_called()

    Patcher.enable_tune_deferred()
    module(torch.ones(1, 2))
    tune.assert_called_once_with()


def test_excluded_constructors_do_not_start_reporting_or_bind_mode(mocker):
    jit_config.mode = JITMode.TUNE_EAGER
    jit_config.patch_exclude = ("unselected_package",)
    start_report = mocker.spy(import_module("aitune.torch.jit.patcher"), "report_tune_run_start")
    Patcher.patch_torch()

    torch.nn.ModuleList()
    torch.nn.ModuleDict()
    torch.nn.Sequential()
    _module_with_module_name("unselected_package.child")

    assert Patcher._session_mode is None
    assert Patcher._session_report is None
    assert Patcher._patched_modules == []
    assert Patcher.intercepted_classes() == []
    assert not has_active_report()
    start_report.assert_not_called()

    jit_config.mode = JITMode.TUNE_DEFERRED
    torch.nn.Linear(2, 2)
    assert Patcher._session_mode is JITMode.TUNE_DEFERRED
    start_report.assert_called_once()


@pytest.mark.parametrize(
    ("initial_mode", "exit_mode"),
    [(JITMode.TUNE_EAGER, JITMode.INSPECT), (JITMode.INSPECT, JITMode.TUNE_DEFERRED)],
)
def test_unstarted_exit_dispatch_uses_current_mode(mocker, initial_mode, exit_mode):
    inspection_exit = mocker.patch.object(InspectModule, "on_python_exit")
    tuning_exit = mocker.patch.object(PatchedModule, "on_python_exit")
    jit_config.mode = initial_mode
    Patcher.patch_torch()
    jit_config.mode = exit_mode

    Patcher._exit_handler()

    if exit_mode is JITMode.INSPECT:
        inspection_exit.assert_called_once_with()
        tuning_exit.assert_not_called()
    else:
        tuning_exit.assert_called_once_with()
        inspection_exit.assert_not_called()
    assert Patcher._session_mode is None
    assert Patcher._session_report is None
    assert not has_active_report()


def test_constructor_interception_rejects_mode_change_after_registering_module():
    jit_config.mode = JITMode.TUNE_EAGER
    Patcher.patch_torch()
    registered = torch.nn.Linear(2, 2)

    jit_config.mode = JITMode.TUNE_DEFERRED
    with pytest.raises(RuntimeError, match=r"Call jit_reset\(\) before switching JIT modes"):
        torch.nn.Linear(2, 2)

    assert Patcher._session_mode is JITMode.TUNE_EAGER
    assert [wrapper.__wrapped__ for wrapper in Patcher._patched_modules] == [registered]


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
