# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for JIT session reporting and incomplete teardown."""

from importlib import import_module

import pytest
import torch

from aitune.torch import jit_reset, register_for_jit_tuning
from aitune.torch.config import AITuneMode
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.config import config as jit_config
from aitune.torch.jit.patched_module import ModuleState, PatchedModule
from aitune.torch.jit.patcher import Patcher
from aitune.torch.tune_data.reporting import (
    has_active_report,
    report_module_tune,
    report_tune_run_end,
    report_tune_run_start,
)
from aitune.utils.disk_space import DiskSpaceError


@pytest.fixture(autouse=True)
def deferred_cpu_session(mocker):
    jit_config.mode = JITMode.TUNE_DEFERRED
    jit_config.device = torch.device("cpu")
    jit_config.min_parameters = -1
    mocker.patch("aitune.torch.tune_data.reporting._flush_active_report")


class _RejectForwardRestore(torch.nn.Linear):
    def __setattr__(self, name, value):
        if name == "forward" and getattr(self, "reject_restore", False):
            raise RuntimeError("forward restoration rejected")
        super().__setattr__(name, value)


def test_reset_preserves_borrowed_report_and_subsequent_spans():
    report = report_tune_run_start(AITuneMode.DECLARATIVE)
    with report_module_tune("before", 1):
        pass

    register_for_jit_tuning([torch.nn.Linear(2, 2)])
    assert Patcher._session_report is None
    jit_reset()

    assert has_active_report()
    with report_module_tune("after", 1):
        pass
    assert [module.module_name for module in report.modules] == ["before", "after"]
    assert report.duration_s is None
    report_tune_run_end(expected_report=report)
    assert not has_active_report()


@pytest.mark.parametrize("finalizer", ["reset", "exit"])
def test_session_finalizers_preserve_replacement_report(finalizer):
    register_for_jit_tuning([torch.nn.Linear(2, 2)])
    owned_report = Patcher._session_report
    replacement = report_tune_run_start(AITuneMode.DECLARATIVE)

    if finalizer == "reset":
        jit_reset()
    else:
        Patcher._exit_handler()

    assert has_active_report()
    assert Patcher._session_report is None
    assert owned_report.duration_s is None
    with report_module_tune("replacement", 1):
        pass
    assert [module.module_name for module in replacement.modules] == ["replacement"]


@pytest.mark.parametrize("existing_report", [False, True])
def test_failed_registration_only_finalizes_report_it_created(mocker, existing_report):
    borrowed = report_tune_run_start(AITuneMode.DECLARATIVE) if existing_report else None
    created_reports = []
    reporting = import_module("aitune.torch.tune_data.reporting")

    def start(mode):
        report = reporting.report_tune_run_start(mode)
        created_reports.append(report)
        return report

    mocker.patch("aitune.torch.jit.patcher.report_tune_run_start", side_effect=start)
    original_register = Patcher._register_module
    modules = [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]

    def register(module, **kwargs):
        if module is modules[1]:
            raise RuntimeError("registration failed")
        return original_register(module, **kwargs)

    mocker.patch.object(Patcher, "_register_module", side_effect=register)
    with pytest.raises(RuntimeError, match="registration failed"):
        register_for_jit_tuning(modules)

    assert Patcher._patched_modules == []
    assert Patcher._session_mode is None
    assert Patcher._session_report is None
    assert has_active_report() is existing_report
    if existing_report:
        assert created_reports == []
        assert borrowed.duration_s is None
        with report_module_tune("still active", 1):
            pass
        assert borrowed.modules[-1].module_name == "still active"
    else:
        assert len(created_reports) == 1
        assert created_reports[0].duration_s is not None
        assert created_reports[0].exception.message == "registration failed"


def test_failed_extension_preserves_existing_session_report(mocker):
    register_for_jit_tuning([torch.nn.Linear(2, 2)])
    owned_report = Patcher._session_report
    handler = Patcher._exit_handler
    mocker.patch.object(Patcher, "_register_module", side_effect=RuntimeError("registration failed"))

    with pytest.raises(RuntimeError, match="registration failed"):
        register_for_jit_tuning([torch.nn.Linear(2, 2)])

    assert Patcher._session_report is owned_report
    assert Patcher._exit_handler is handler
    assert owned_report.duration_s is None
    jit_reset()
    assert owned_report.duration_s is not None
    assert not has_active_report()


def test_failed_registration_preserves_original_error_when_report_flush_fails(mocker):
    mocker.patch.object(Patcher, "_register_module", side_effect=RuntimeError("registration failed"))
    mocker.patch(
        "aitune.torch.tune_data.reporting._flush_active_report",
        side_effect=DiskSpaceError(message="report disk full"),
    )

    with pytest.raises(RuntimeError, match="registration failed") as raised:
        register_for_jit_tuning([torch.nn.Linear(2, 2)])

    assert "report finalization failed" in raised.value.__notes__[0]
    assert "No space left on AITune cache device" in raised.value.__notes__[0]
    assert Patcher._session_mode is None
    assert Patcher._session_report is None
    assert Patcher._exit_handler is None
    assert not has_active_report()


def test_partial_reset_blocks_registration_until_cleanup_retry_succeeds():
    good = torch.nn.Linear(2, 2)
    failing = _RejectForwardRestore(2, 2)
    registration = register_for_jit_tuning([good, failing])
    wrappers = list(Patcher._patched_modules)
    counter = PatchedModule.module_counter
    failing.reject_restore = True
    try:
        with pytest.raises(RuntimeError, match="forward restoration rejected"):
            jit_reset()

        assert Patcher._patched_modules == [wrappers[1]]
        assert Patcher._cleanup_pending
        assert Patcher._session_mode == JITMode.TUNE_DEFERRED
        assert PatchedModule.module_counter == counter
        assert registration.state_counts == {"detached": 2}
        for module in (failing, torch.nn.Linear(2, 2)):
            with pytest.raises(RuntimeError, match="cleanup is incomplete"):
                register_for_jit_tuning([module])
        with pytest.raises(RuntimeError, match="cleanup is incomplete"):
            Patcher.patch_torch()
    finally:
        failing.reject_restore = False
        jit_reset()

    assert Patcher._patched_modules == []
    assert not Patcher._cleanup_pending
    assert Patcher._session_mode is None
    assert PatchedModule.module_counter == 0
    fresh = register_for_jit_tuning([failing])
    assert fresh.state_counts == {"init": 1}


def test_failed_rollback_preserves_cleanup_retry_and_registration_error(mocker):
    failing = _RejectForwardRestore(2, 2)
    good = torch.nn.Linear(2, 2)
    rejected = torch.nn.Linear(2, 2)
    original_register = Patcher._register_module

    def register(module, **kwargs):
        if module is rejected:
            failing.reject_restore = True
            raise RuntimeError("registration failed")
        return original_register(module, **kwargs)

    mocker.patch.object(Patcher, "_register_module", side_effect=register)
    try:
        with pytest.raises(RuntimeError, match="registration failed") as raised:
            register_for_jit_tuning([good, failing, rejected])

        assert "forward restoration rejected" in raised.value.__notes__[0]
        assert len(Patcher._patched_modules) == 1
        assert Patcher._patched_modules[0].__wrapped__ is failing
        assert Patcher._patched_modules[0]._state == ModuleState.DETACHED
        assert good.forward.__func__ is torch.nn.Linear.forward
        assert Patcher._cleanup_pending
        assert Patcher._session_mode == JITMode.TUNE_DEFERRED
        assert PatchedModule.module_counter == 2
        assert not has_active_report()
        with pytest.raises(RuntimeError, match="cleanup is incomplete"):
            register_for_jit_tuning([failing])
    finally:
        failing.reject_restore = False
        jit_reset()

    assert not Patcher._cleanup_pending
    assert Patcher._patched_modules == []
    assert PatchedModule.module_counter == 0
