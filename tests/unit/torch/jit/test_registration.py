# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for explicitly registering existing modules for JIT tuning."""

from unittest.mock import Mock

import pytest
import torch

import aitune.torch as aitune_torch
from aitune.torch import JITRegistration, jit_reset, register_for_jit_tuning
from aitune.torch.backend.torch_eager import TorchEagerBackend
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.config import config as jit_config
from aitune.torch.jit.patched_module import ModuleState, PatchedModule
from aitune.torch.jit.patcher import Patcher, prepare_for_jit_tuning
from aitune.torch.jit.tune import deferred as jit_deferred
from aitune.torch.tune_data.reporting import has_active_report
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


def _configure_dry_run(tmp_path, strategy: Mock) -> None:
    _configure_deferred(tmp_path)
    jit_config.dry_run = True
    jit_config.dry_run_failure_probability = 0.0
    jit_config.strategy = strategy


def _strategy() -> Mock:
    strategy = Mock()
    strategy.clone.return_value = strategy
    strategy.backend_results = []
    return strategy


def _settled_registration(tmp_path, state, module=None):
    strategy = None
    if state is ModuleState.EAGER:
        strategy = _strategy()
        strategy.tune_dry_run.side_effect = RuntimeError("forced tuning failure")
        _configure_dry_run(tmp_path, strategy)
    else:
        _configure_deferred(tmp_path)
        jit_config.min_parameters = 100
    if module is None:
        module = torch.nn.Linear(2, 2)
    registration = register_for_jit_tuning([module])
    wrapper = Patcher._patched_modules[0]
    module(torch.ones(1, 2))
    if state is ModuleState.EAGER:
        Patcher.enable_tune_deferred()
        module(torch.ones(1, 2))
        strategy.tune_dry_run.assert_called_once()
    assert registration.state_counts == {state.value: 1}
    assert Patcher._patched_modules == []
    return module, registration, wrapper, strategy


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


def test_registration_deduplicates_modules_by_identity_and_is_idempotent(tmp_path):
    _configure_deferred(tmp_path)
    module = torch.nn.Linear(2, 2)

    first = register_for_jit_tuning([module, module])
    second = register_for_jit_tuning([module])

    assert first.modules == second.modules == (module,)
    assert len(Patcher._patched_modules) == 1
    assert first.reports[0].module_id == second.reports[0].module_id


@pytest.mark.parametrize("state", [ModuleState.SKIPPED, ModuleState.EAGER])
def test_repeated_registration_reuses_settled_target_without_retry(tmp_path, state):
    module, first, wrapper, strategy = _settled_registration(tmp_path, state)
    original_forward = module.forward
    module_counter = PatchedModule.module_counter
    heads = list(PatchedModule.heads)
    expected = module(torch.ones(1, 2))

    second = register_for_jit_tuning([module, module])
    torch.testing.assert_close(module(torch.ones(1, 2)), expected)

    assert second.modules == first.modules == (module,)
    assert first._patched_modules == second._patched_modules == (wrapper,)
    assert first.reports[0].module_id == second.reports[0].module_id
    assert second.state_counts == {state.value: 1}
    assert module.forward is original_forward
    assert Patcher._patched_modules == []
    assert PatchedModule.module_counter == module_counter
    assert PatchedModule.heads == heads
    if strategy is not None:
        strategy.tune_dry_run.assert_called_once()


@pytest.mark.parametrize("state", [ModuleState.SKIPPED, ModuleState.EAGER])
def test_settled_target_retains_ownership_and_blocks_constructor_interception(tmp_path, state):
    parent = _ParentModule()
    module, registration, wrapper, _ = _settled_registration(tmp_path, state, parent.child)
    original_module_init = torch.nn.Module.__init__

    with pytest.raises(ValueError, match="overlaps an existing JIT registration"):
        register_for_jit_tuning([parent])
    with pytest.raises(RuntimeError, match="explicit JIT registrations are active"):
        Patcher.patch_torch()

    assert registration.state_counts == {state.value: 1}
    assert Patcher._explicit_registrations == {id(module): wrapper}
    assert Patcher._patched_modules == []
    assert not Patcher._torch_patched
    assert torch.nn.Module.__init__ is original_module_init
    assert "forward" not in parent.__dict__


@pytest.mark.parametrize("state", [ModuleState.SKIPPED, ModuleState.EAGER])
def test_failed_mixed_registration_preserves_settled_target_and_rolls_back_new_targets(tmp_path, mocker, state):
    module, first, wrapper, strategy = _settled_registration(tmp_path, state)
    added = torch.nn.Linear(2, 2)
    rejected = torch.nn.Linear(2, 2)
    original_register = Patcher._register_module
    history = dict(Patcher._explicit_registrations)
    module_counter = PatchedModule.module_counter
    report = Patcher._session_report
    created = []

    def register(target, **kwargs):
        if target is rejected:
            raise RuntimeError("registration failed")
        new_wrapper = original_register(target, **kwargs)
        created.append(new_wrapper)
        return new_wrapper

    register_mock = mocker.patch.object(Patcher, "_register_module", side_effect=register)
    with pytest.raises(RuntimeError, match="registration failed"):
        register_for_jit_tuning([module, added, rejected])

    assert register_mock.call_count == 2
    assert len(created) == 1
    assert created[0].__wrapped__ is added
    assert added.forward is created[0]._original_forward
    assert "forward" not in rejected.__dict__
    assert Patcher._explicit_registrations == history
    assert Patcher._patched_modules == []
    assert not Patcher._cleanup_pending
    assert PatchedModule.module_counter == module_counter
    assert Patcher._session_report is report
    assert first.state_counts == {state.value: 1}
    assert register_for_jit_tuning([module])._patched_modules == (wrapper,)
    if strategy is not None:
        strategy.tune_dry_run.assert_called_once()

    replacement = register_for_jit_tuning([added])
    assert replacement.state_counts == {"init": 1}
    assert replacement.reports[0].module_id == module_counter
    assert set(Patcher._explicit_registrations) == {id(module), id(added)}


@pytest.mark.parametrize("state", [ModuleState.SKIPPED, ModuleState.EAGER])
def test_reset_detaches_settled_handle_and_allows_a_fresh_registration_generation(tmp_path, state):
    module, first, wrapper, _ = _settled_registration(tmp_path, state)

    jit_reset()

    assert first.state_counts == {"detached": 1}
    assert Patcher._explicit_registrations == {}
    assert Patcher._patched_modules == []
    assert Patcher._session_mode is None
    assert not has_active_report()

    second = register_for_jit_tuning([module])
    assert second.state_counts == {"init": 1}
    assert second.reports[0].module_id == 0
    assert second._patched_modules[0] is not wrapper
    assert first.state_counts == {"detached": 1}
    assert set(Patcher._explicit_registrations) == {id(module)}


def test_registration_handle_reports_live_explicit_head_state(tmp_path):
    _configure_deferred(tmp_path)
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

    for patched in Patcher._patched_modules:
        patched._state = ModuleState.TUNED

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

    with pytest.raises(RuntimeError, match="explicit JIT registrations"):
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
def test_registration_rejects_switching_tuning_mode_within_active_session(tmp_path, initial_mode, next_mode):
    _configure_deferred(tmp_path)
    jit_config.mode = initial_mode
    first = torch.nn.Linear(2, 2)
    second = torch.nn.Linear(2, 2)
    register_for_jit_tuning([first])

    jit_config.mode = next_mode
    with pytest.raises(RuntimeError, match=r"Call jit_reset\(\) before switching JIT modes"):
        register_for_jit_tuning([second])

    assert Patcher._session_mode is initial_mode
    assert [patched.__wrapped__ for patched in Patcher._patched_modules] == [first]
    assert "forward" not in second.__dict__


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
    with pytest.raises(RuntimeError, match=r"Call jit_reset\(\) before switching JIT modes"):
        module(torch.ones(1, 2))

    patched = Patcher._patched_modules[0]
    assert isinstance(patched, PatchedModule)
    assert patched._state is ModuleState.INIT


def test_registration_rejects_an_active_automatic_jit_session(tmp_path):
    _configure_deferred(tmp_path)
    with prepare_for_jit_tuning():
        automatic = torch.nn.Linear(2, 2)
    explicit = torch.nn.Linear(2, 2)

    with pytest.raises(RuntimeError, match=r"automatic JIT discovery; call jit_reset\(\) first"):
        register_for_jit_tuning([explicit])

    assert len(Patcher._patched_modules) == 1
    assert Patcher._patched_modules[0].__wrapped__ is automatic
    assert "forward" not in explicit.__dict__

    jit_reset()

    registration = register_for_jit_tuning([explicit])
    assert registration.modules == (explicit,)


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


def test_registration_rejects_overlap_with_an_existing_explicit_registration(tmp_path):
    _configure_deferred(tmp_path)
    parent = _ParentModule()
    registration = register_for_jit_tuning([parent])
    installed_forward = parent.forward

    with pytest.raises(ValueError, match="overlaps an existing JIT registration"):
        register_for_jit_tuning([parent.child])

    assert registration.modules == (parent,)
    assert parent.forward is installed_forward
    assert "forward" not in parent.child.__dict__
    assert [patched.__wrapped__ for patched in Patcher._patched_modules] == [parent]


def test_registration_validates_every_target_before_patching_any_target(tmp_path):
    _configure_deferred(tmp_path)
    valid = torch.nn.Linear(2, 2)
    excluded = torch.nn.Sequential(torch.nn.Linear(2, 2))

    with pytest.raises(ValueError, match="excluded"):
        register_for_jit_tuning([valid, excluded])

    assert "forward" not in valid.__dict__
    assert Patcher._patched_modules == []


def test_registration_rolls_back_partial_wrapper_and_new_session_on_failure(tmp_path, mocker):
    _configure_deferred(tmp_path)
    modules = [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]
    original_register_module = Patcher._register_module
    call_count = 0

    def register_module(module, *, explicit_head=False):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("registration failed")
        return original_register_module(module, explicit_head=explicit_head)

    mocker.patch.object(Patcher, "_register_module", side_effect=register_module)

    with pytest.raises(RuntimeError, match="registration failed"):
        register_for_jit_tuning(modules)

    assert Patcher._patched_modules == []
    assert Patcher._session_mode is None
    assert Patcher._exit_handler is None
    assert PatchedModule.module_counter == 0
    assert not has_active_report()
    assert all(module.forward.__func__ is torch.nn.Linear.forward for module in modules)


def test_jit_reset_restores_all_targets_and_starts_a_clean_registration_session(tmp_path):
    _configure_deferred(tmp_path)
    modules = [torch.nn.Linear(2, 2) for _ in range(3)]
    first = register_for_jit_tuning(modules)
    first_handler = Patcher._exit_handler

    assert [report.module_id for report in first.reports] == [0, 1, 2]
    assert has_active_report()

    jit_reset()

    assert Patcher._patched_modules == []
    assert PatchedModule.heads == []
    assert PatchedModule.module_counter == 0
    assert Patcher._exit_handler is None
    assert not has_active_report()
    assert all(module.forward.__func__ is torch.nn.Linear.forward for module in modules)
    assert first.state_counts == {"detached": 3}
    assert not first.all_tuned

    replacement = torch.nn.Linear(2, 2)
    second = register_for_jit_tuning([replacement])

    assert second.reports[0].module_id == 0
    assert Patcher._exit_handler is not None
    assert Patcher._exit_handler is first_handler
    assert has_active_report()


def test_jit_reset_restores_aitune_generated_device_class(tmp_path):
    strategy = _strategy()
    _configure_dry_run(tmp_path, strategy)
    module = torch.nn.Linear(2, 2)
    original_class = module.__class__
    register_for_jit_tuning([module])

    module(torch.ones(1, 2))
    Patcher.enable_tune_deferred()
    module(torch.ones(1, 2))
    assert module.__class__ is not original_class

    jit_reset()

    assert module.__class__ is original_class
    assert module.forward.__func__ is torch.nn.Linear.forward


def test_registration_api_and_jit_reset_are_exported_from_the_public_torch_api():
    assert aitune_torch.register_for_jit_tuning is register_for_jit_tuning
    assert aitune_torch.JITRegistration is JITRegistration
    assert aitune_torch.jit_reset is jit_reset
    assert "register_for_jit_tuning" in aitune_torch.__all__
    assert "JITRegistration" in aitune_torch.__all__
    assert "jit_reset" in aitune_torch.__all__
