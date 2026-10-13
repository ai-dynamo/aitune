# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for explicitly registering existing modules for JIT tuning."""

import json
from importlib import import_module

import pytest
import torch

import aitune.torch as aitune_torch
from aitune.torch import JITRegistration, register_for_jit_tuning
from aitune.torch.backend.torch_eager import TorchEagerBackend
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.config import config as jit_config
from aitune.torch.jit.patched_module import ModuleState, PatchedModule
from aitune.torch.jit.patcher import Patcher, prepare_for_jit_tuning
from aitune.torch.jit.tune import deferred as jit_deferred
from aitune.torch.tune_data.reporting import report_tune_run_end
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy


class _ParentModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.child = torch.nn.Linear(2, 2)

    def forward(self, value):
        return self.child(value)


class _NonOwningCaller(torch.nn.Module):
    """Call another module without adding it to this module's ownership tree."""

    def __init__(self, callee: torch.nn.Module):
        super().__init__()
        object.__setattr__(self, "callee", callee)

    def forward(self, value):
        return self.callee(value)


def _configure_deferred(tmp_path) -> None:
    jit_config.mode = JITMode.TUNE_DEFERRED
    jit_config.device = torch.device("cpu")
    jit_config.batch_axis_required = False
    jit_config.min_parameters = -1
    jit_config.cache_dir = tmp_path / "jit-cache"


def test_registers_only_existing_modules_selected_by_the_caller(tmp_path):
    _configure_deferred(tmp_path)
    selected = torch.nn.Linear(2, 2)
    unrelated = torch.nn.Linear(2, 2)
    original_module_init = torch.nn.Module.__init__

    registration = register_for_jit_tuning([selected])

    assert isinstance(registration, JITRegistration)
    assert registration.modules == (selected,)
    assert "forward" in selected.__dict__
    assert "forward" not in unrelated.__dict__
    assert torch.nn.Module.__init__ is original_module_init
    assert [patched.__wrapped__ for patched in Patcher._patched_modules] == [selected]


def test_module_list_argument_registers_its_elements_not_the_container(tmp_path):
    _configure_deferred(tmp_path)
    modules = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])

    registration = register_for_jit_tuning(modules)

    assert registration.modules == tuple(modules)
    assert "forward" not in modules.__dict__
    assert all("forward" in module.__dict__ for module in modules)


def test_single_module_requires_an_iterable_and_generator_is_consumed_once(tmp_path):
    _configure_deferred(tmp_path)
    module = torch.nn.Linear(2, 2)

    with pytest.raises(TypeError, match=r"\[module\]"):
        register_for_jit_tuning(module)

    yielded = []

    def modules():
        yielded.append(module)
        yield module

    registration = register_for_jit_tuning(modules())

    assert yielded == [module]
    assert registration.modules == (module,)


def test_registration_deduplicates_one_batch_and_rejects_another_batch(tmp_path):
    _configure_deferred(tmp_path)
    module = torch.nn.Linear(2, 2)
    registration = register_for_jit_tuning([module, module])
    installed_forward = module.forward
    additional = torch.nn.Linear(2, 2)

    for targets in ([module], [additional]):
        with pytest.raises(RuntimeError, match="fresh worker"):
            register_for_jit_tuning(targets)

    assert registration.modules == (module,)
    assert len(Patcher._patched_modules) == 1
    assert module.forward is installed_forward
    assert "forward" not in additional.__dict__


def test_registration_handle_reports_live_explicit_head_state(tmp_path):
    _configure_deferred(tmp_path)
    jit_config.strategy = OneBackendStrategy(TorchEagerBackend()).enable_performance_validation(False)
    modules = [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]
    registration = register_for_jit_tuning(modules)

    assert registration.state_counts == {"init": 2}
    assert not registration.all_tuned

    for module in modules:
        module(torch.ones(1, 2))

    reports = registration.reports
    assert registration.state_counts == {"recording": 2}
    assert not registration.all_tuned
    assert [report.level for report in reports] == [0, 0]
    assert [report.parent_module_id for report in reports] == [None, None]
    assert [report.allowed_to_tune for report in reports] == [True, True]
    assert [report.call_count for report in reports] == [1, 1]

    jit_deferred()
    for module in modules:
        module(torch.ones(1, 2))

    assert registration.state_counts == {"tuned": 2}
    assert registration.all_tuned


def test_registration_handle_reports_modules_filtered_by_min_parameters_as_skipped(tmp_path):
    _configure_deferred(tmp_path)
    jit_config.min_parameters = 100
    module = torch.nn.Linear(2, 2)
    registration = register_for_jit_tuning([module])

    module(torch.ones(1, 2))

    assert registration.state_counts == {"skipped": 1}
    assert not registration.all_tuned
    report = registration.reports[0]
    assert report.level == 0
    assert report.call_count == 1
    assert report.module_name == "Linear"
    assert report.parent_module_id is None
    assert not report.allowed_to_tune
    assert Patcher._patched_modules == []
    assert module.forward.__func__ is torch.nn.Linear.forward


def test_explicit_targets_remain_independent_heads_in_a_nested_runtime_call(tmp_path):
    _configure_deferred(tmp_path)
    callee = torch.nn.Linear(2, 2)
    caller = _NonOwningCaller(callee)
    registration = register_for_jit_tuning([caller, callee])

    caller(torch.ones(1, 2))

    assert set(PatchedModule.heads) == set(Patcher._patched_modules)
    assert len(PatchedModule.heads) == 2
    assert all(report.level == 0 for report in registration.reports)
    assert all(report.parent_module_id is None for report in registration.reports)


def test_explicit_registration_uses_existing_eager_tuning_lifecycle(tmp_path, mocker):
    _configure_deferred(tmp_path)
    jit_config.mode = JITMode.TUNE_EAGER
    module = torch.nn.Linear(2, 2)
    tune = mocker.patch.object(PatchedModule, "tune")
    register_for_jit_tuning([module])

    module(torch.ones(1, 2))

    tune.assert_called_once_with()


def test_explicit_registration_deferred_tuning_preserves_all_recorded_routes(tmp_path):
    _configure_deferred(tmp_path)
    jit_config.strategy = OneBackendStrategy(TorchEagerBackend()).enable_performance_validation(False)
    module = torch.nn.Linear(2, 2).eval()
    registration = register_for_jit_tuning([module])
    vector = torch.tensor([1.0, 2.0])
    matrix = torch.tensor([[1.0, 2.0]])

    with torch.inference_mode():
        expected_vector = module(vector)
        expected_matrix = module(matrix)

        assert registration.state_counts == {"recording": 1}
        assert len(registration.reports[0].graphs) == 2

        jit_deferred()
        module(vector)

        assert registration.all_tuned
        torch.testing.assert_close(module(vector), expected_vector)
        torch.testing.assert_close(module(matrix), expected_matrix)


def test_prepared_units_keep_identity_forward_hooks_and_reports_after_tuning(tmp_path, monkeypatch):
    """Provider-owned prepared forwards are usable without a FlexTensor dependency."""
    _configure_deferred(tmp_path)
    jit_config.detect_graph_breaks = False
    jit_config.strategy = OneBackendStrategy(TorchEagerBackend()).enable_performance_validation(False)
    report_path = tmp_path / "report.json"
    reporting = import_module("aitune.torch.tune_data.reporting")
    monkeypatch.setattr(reporting.config, "tuning_data_output_path", report_path)
    modules = [torch.nn.Linear(2, 2).eval(), torch.nn.Linear(2, 2).eval()]
    traces = [[], []]

    def prepare(module, trace):
        original = module.forward

        def prepared(value):
            trace.append("enter")
            result = original(value) + 7
            trace.append("leave")
            return result

        module.forward = prepared
        module.register_forward_pre_hook(lambda *_: trace.append("pre-hook"))
        module.register_forward_hook(lambda *_: trace.append("post-hook"))
        return prepared

    forwards = [prepare(module, trace) for module, trace in zip(modules, traces, strict=True)]
    value = torch.ones(1, 2)
    expected = [module(value) for module in modules]
    registration = register_for_jit_tuning(modules)
    assert registration.modules == tuple(modules)
    assert [wrapper._original_forward for wrapper in registration._patched_modules] == forwards

    with torch.inference_mode():
        for module in modules:
            module(value)
        jit_deferred()
        for module in modules:
            module(value)
        assert registration.all_tuned
        for module, trace, output in zip(modules, traces, expected, strict=True):
            trace.clear()
            torch.testing.assert_close(module(value), output)
            assert trace == ["pre-hook", "enter", "leave", "post-hook"]

    assert [report.module_id for report in registration.reports] == [0, 1]
    report_tune_run_end()
    telemetry = json.loads(report_path.read_text())
    assert telemetry["mode"] == "JIT"
    assert len(telemetry["modules"]) == 2
    assert all("TorchEagerBackend" in report["graphs"][0]["selected_backend"] for report in telemetry["modules"])


@pytest.mark.parametrize("modules", [[], [object()]])
def test_registration_rejects_invalid_module_collections_without_mutation(modules, tmp_path):
    _configure_deferred(tmp_path)

    with pytest.raises((TypeError, ValueError)):
        register_for_jit_tuning(modules)

    assert Patcher._patched_modules == []


def test_registration_rejects_inspection_mode_without_mutation(tmp_path):
    _configure_deferred(tmp_path)
    jit_config.mode = JITMode.INSPECT
    module = torch.nn.Linear(2, 2)

    with pytest.raises(RuntimeError, match="INSPECT"):
        register_for_jit_tuning([module])

    assert "forward" not in module.__dict__
    assert Patcher._patched_modules == []


def test_registration_rejects_active_constructor_interception_without_mutation(tmp_path):
    _configure_deferred(tmp_path)
    module = torch.nn.Linear(2, 2)

    with prepare_for_jit_tuning(), pytest.raises(RuntimeError):
        register_for_jit_tuning([module])

    assert "forward" not in module.__dict__
    assert Patcher._patched_modules == []


def test_constructor_interception_rejects_an_active_explicit_registration(tmp_path):
    _configure_deferred(tmp_path)
    module = torch.nn.Linear(2, 2)
    register_for_jit_tuning([module])

    with pytest.raises(RuntimeError, match="fresh worker"):
        with prepare_for_jit_tuning():
            pass

    assert not Patcher._torch_patched
    assert len(Patcher._patched_modules) == 1


@pytest.mark.parametrize(
    ("initial_mode", "next_mode"),
    [
        (JITMode.TUNE_EAGER, JITMode.TUNE_DEFERRED),
        (JITMode.TUNE_DEFERRED, JITMode.TUNE_EAGER),
    ],
)
def test_registered_module_forward_rejects_switching_tuning_mode(tmp_path, initial_mode, next_mode):
    _configure_deferred(tmp_path)
    jit_config.mode = initial_mode
    module = torch.nn.Linear(2, 2)
    register_for_jit_tuning([module])

    jit_config.mode = next_mode
    with pytest.raises(RuntimeError, match=r"fresh worker"):
        module(torch.ones(1, 2))

    patched = Patcher._patched_modules[0]
    assert isinstance(patched, PatchedModule)
    assert patched._state is ModuleState.INIT


def test_registration_rejects_an_active_automatic_jit_session(tmp_path):
    _configure_deferred(tmp_path)
    with prepare_for_jit_tuning():
        automatic = torch.nn.Linear(2, 2)
    explicit = torch.nn.Linear(2, 2)

    with pytest.raises(RuntimeError, match="fresh worker"):
        register_for_jit_tuning([explicit])

    assert len(Patcher._patched_modules) == 1
    assert Patcher._patched_modules[0].__wrapped__ is automatic
    assert "forward" not in explicit.__dict__


def test_registration_rejects_excluded_exact_target_without_mutation(tmp_path):
    _configure_deferred(tmp_path)
    excluded = torch.nn.Sequential(torch.nn.Linear(2, 2))

    with pytest.raises(ValueError, match="excluded"):
        register_for_jit_tuning([excluded])

    assert "forward" not in excluded.__dict__
    assert "forward" not in excluded[0].__dict__
    assert Patcher._patched_modules == []


def test_registration_rejects_overlapping_ownership_trees_atomically(tmp_path):
    _configure_deferred(tmp_path)
    parent = _ParentModule()

    with pytest.raises(ValueError, match="disjoint"):
        register_for_jit_tuning([parent, parent.child])

    assert "forward" not in parent.__dict__
    assert "forward" not in parent.child.__dict__
    assert Patcher._patched_modules == []

    other = _ParentModule()
    other.child = parent.child
    with pytest.raises(ValueError, match="disjoint"):
        register_for_jit_tuning([parent, other])
    registration = register_for_jit_tuning([parent])
    assert registration.modules == (parent,)


def test_registration_validates_every_target_before_patching_any_target(tmp_path):
    _configure_deferred(tmp_path)
    valid = torch.nn.Linear(2, 2)
    excluded = torch.nn.Sequential(torch.nn.Linear(2, 2))

    with pytest.raises(ValueError, match="excluded"):
        register_for_jit_tuning([valid, excluded])

    assert "forward" not in valid.__dict__
    assert Patcher._patched_modules == []


def test_partial_installation_is_terminal_and_keeps_original_error(tmp_path, monkeypatch):
    _configure_deferred(tmp_path)
    first, second = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    original_register = Patcher._register_module
    failure = RuntimeError("registration failed")

    def register(module, **kwargs):
        if module is second:
            raise failure
        return original_register(module, **kwargs)

    monkeypatch.setattr(Patcher, "_register_module", register)
    with pytest.raises(RuntimeError) as raised:
        register_for_jit_tuning([first, second])

    assert raised.value is failure
    assert "forward" in first.__dict__
    assert "forward" not in second.__dict__
    assert [wrapper.__wrapped__ for wrapper in Patcher._patched_modules] == [first]
    with pytest.raises(RuntimeError, match="fresh worker"):
        register_for_jit_tuning([second])


def test_materializing_targets_cannot_hide_automatic_activation(tmp_path):
    _configure_deferred(tmp_path)

    def targets():
        with prepare_for_jit_tuning():
            yield torch.nn.Linear(2, 2)

    with pytest.raises(RuntimeError, match="fresh worker"):
        register_for_jit_tuning(targets())

    assert len(Patcher._patched_modules) == 1
    assert Patcher._entry_path == "automatic"


def test_registration_public_api_has_no_reset():
    assert aitune_torch.register_for_jit_tuning is register_for_jit_tuning
    assert aitune_torch.JITRegistration is JITRegistration
    assert "register_for_jit_tuning" in aitune_torch.__all__
    assert "JITRegistration" in aitune_torch.__all__
    assert not hasattr(aitune_torch, "jit_reset")
    assert "jit_reset" not in aitune_torch.__all__
