# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public-manager adapter tests without a FlexTensor installation or GPU."""

from types import SimpleNamespace

import pytest
import torch
import wrapt

from aitune.torch.backend.torch_eager import TorchEagerBackend
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.config import config as jit_config
from aitune.torch.jit.flextensor import (
    FlexTensorUnit,
    _tuned_forward_signature,
    register_flextensor_units,
    replan_flextensor,
)
from aitune.torch.jit.patched_module import PatchedModule
from aitune.torch.jit.tune import deferred as jit_deferred
from aitune.torch.module.tuned_module import TunedModule
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy
from aitune.torch.utils.module import is_externally_managed_module
from tests.toy_backends import SleepBackend


class _Manager:
    """Expose only the public provider contract consumed by the adapter."""

    def __init__(self, modules=None, manager_id=1):
        modules = [torch.nn.Linear(2, 2).eval()] if modules is None else modules
        modules = [module if isinstance(module, FlexTensorUnit) else FlexTensorUnit(module) for module in modules]
        self.model = torch.nn.Sequential(*modules).eval()
        self.units = tuple((str(index), module) for index, module in enumerate(modules))
        self.compiled_offload_manager_id = manager_id
        self.phase = SimpleNamespace(value="inference")
        self.external_compile_active = True
        self.compiled_offload_active = True
        self.compiled_replan_active = True
        self.compiled_profile_active = False
        self.strategy_replan_status = SimpleNamespace(state="idle", can_request=True, applied=None)
        self.discovery_calls = 0
        self.requests = 0
        self.forward_calls = 0
        self.requested_count = 2
        self.apply = True

    def offload_units(self):
        self.discovery_calls += 1
        return self.units

    def request_strategy_replan(self):
        self.requests += 1
        self.strategy_replan_status = SimpleNamespace(state="in_progress", can_request=False, applied=None)
        return self.requested_count

    def forward(self):
        with torch.inference_mode():
            value = self.model(torch.ones(1, 2))
        self.forward_calls += 1
        if self.forward_calls == self.requested_count:
            self.strategy_replan_status = SimpleNamespace(state="done", can_request=False, applied=self.apply)
        return value


class _ParentModule(torch.nn.Module):
    def __init__(self, child):
        super().__init__()
        self.child = child

    def forward(self, value):
        return self.child(value)


@pytest.fixture(autouse=True)
def _configure_jit():
    jit_config.mode = JITMode.TUNE_DEFERRED
    jit_config.device = torch.device("cpu")
    jit_config.batch_axis_required = False
    jit_config.min_parameters = -1
    jit_config.strategy = OneBackendStrategy(TorchEagerBackend()).enable_performance_validation(False)


def _settle(manager, registration):
    with torch.inference_mode():
        expected = manager.model(torch.ones(1, 2))
        if jit_config.mode == JITMode.TUNE_DEFERRED:
            jit_deferred()
        actual = manager.model(torch.ones(1, 2))
        torch.testing.assert_close(actual, expected)
    assert registration.all_tuned


def test_unit_preserves_original_kwargs_structured_outputs_hooks_and_training():
    events = []

    class Compute(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 2).eval()

        def forward(self, value, *, scale, marker):
            events.append("compute")
            return {"value": self.layer(value) * scale, "marker": marker}

    compute = Compute()
    parameter = compute.layer.weight
    compute.register_forward_pre_hook(lambda *args: events.append("compute_pre"))
    compute.register_forward_hook(lambda *args: events.append("compute_post"))
    shell = FlexTensorUnit(compute)
    shell.register_forward_pre_hook(lambda *args: events.append("shell_pre"))
    shell.register_forward_hook(lambda *args: events.append("shell_post"))
    marker = object()

    result = shell(torch.ones(1, 2), scale=2, marker=marker)

    assert result["marker"] is marker
    assert events == ["shell_pre", "compute_pre", "compute", "compute_post", "shell_post"]
    assert shell.compute is compute
    assert shell.compute.layer.weight is parameter
    assert shell.training is compute.training is True
    assert compute.layer.training is False
    assert list(shell.state_dict()) == ["compute.layer.weight", "compute.layer.bias"]
    assert FlexTensorUnit(torch.nn.Linear(2, 2).eval()).training is False


@pytest.mark.parametrize("invalid", [None, object(), torch.ones(1)])
def test_unit_requires_a_module(invalid):
    with pytest.raises(TypeError, match="compute must be a torch.nn.Module"):
        FlexTensorUnit(invalid)


@pytest.mark.parametrize("nested", [False, True])
def test_unit_rejects_nested_or_double_wrapping(nested):
    shell = FlexTensorUnit(torch.nn.Linear(2, 2))
    compute = _ParentModule(shell) if nested else shell

    with pytest.raises(ValueError, match="nested or wrapped twice"):
        FlexTensorUnit(compute)


def test_registration_uses_final_live_public_shells_not_discovery_predecessors():
    manager = _Manager()
    predecessor = manager.units[0][1]
    final = FlexTensorUnit(torch.nn.Linear(2, 2))
    manager.model[0] = final
    manager.units = (("final", final),)

    registration = register_flextensor_units([manager])

    assert registration.modules == (final.compute,)
    assert "forward" not in final.__dict__
    assert "forward" not in predecessor.compute.__dict__


def test_unprepared_provider_unit_is_rejected_without_patching_it():
    manager = _Manager()
    plain = torch.nn.Linear(2, 2)
    manager.model = torch.nn.Sequential(plain)
    manager.units = (("plain", plain),)

    with pytest.raises(RuntimeError, match="insert FlexTensorUnit.*before flextensor.offload"):
        register_flextensor_units([manager])

    assert "forward" not in plain.__dict__
    assert not is_externally_managed_module(plain)


def test_compute_must_be_the_live_registered_child_in_the_manager_model():
    manager = _Manager()
    shell = manager.units[0][1]
    orphan = torch.nn.Linear(2, 2)
    object.__setattr__(shell, "compute", orphan)

    with pytest.raises(RuntimeError, match="compute must belong to the manager's current model"):
        register_flextensor_units([manager])

    assert "forward" not in orphan.__dict__


@pytest.mark.parametrize("different_managers", [False, True])
def test_distinct_shells_cannot_share_the_same_compute(different_managers):
    compute = torch.nn.Linear(2, 2)
    shells = [FlexTensorUnit(compute), FlexTensorUnit(compute)]
    managers = (
        [_Manager([shell], manager_id=index) for index, shell in enumerate(shells)]
        if different_managers
        else [_Manager(shells)]
    )

    with pytest.raises(ValueError, match="share a compute"):
        register_flextensor_units(managers)

    assert "forward" not in compute.__dict__


def test_compute_children_are_not_public_subset_selectors():
    manager = _Manager()

    with pytest.raises(ValueError, match="offload_units"):
        register_flextensor_units([manager], modules=[manager.units[0][1].compute])

    assert "forward" not in manager.units[0][1].compute.__dict__


def test_retained_profile_registers_without_replan_capability_and_never_arms_a_tail():
    manager = _Manager()
    manager.external_compile_active = False
    manager.compiled_replan_active = False
    manager.compiled_profile_active = True
    manager.request_strategy_replan = None
    manager.strategy_replan_status = None

    registration = register_flextensor_units([manager])
    _settle(manager, registration)

    with pytest.raises(RuntimeError, match="retained-profile mode cannot replan"):
        replan_flextensor(manager, registration, manager.forward)

    assert manager.requests == manager.forward_calls == 0


@pytest.mark.parametrize("mode", [JITMode.TUNE_EAGER, JITMode.TUNE_DEFERRED])
def test_ordinary_aitune_replays_compute_inside_one_prepared_residency_call(mode):
    manager = _Manager()
    shell = manager.units[0][1]
    shell_forward, compute_forward = shell.forward, shell.compute.forward
    calls = {"enter": 0, "exit": 0, "compute": 0, "resident": False}

    def compute(value):
        assert calls["resident"]
        calls["compute"] += 1
        return compute_forward(value)

    def prepared(value):
        assert not calls["resident"]
        calls["enter"] += 1
        calls["resident"] = True
        try:
            return shell_forward(value)
        finally:
            calls["resident"] = False
            calls["exit"] += 1

    shell.compute.forward = compute
    shell.forward = prepared
    jit_config.mode = mode
    registration = register_flextensor_units([manager])

    _settle(manager, registration)

    assert calls["enter"] == calls["exit"] == 2
    assert calls["compute"] > calls["enter"]
    assert not calls["resident"]
    assert shell.forward is prepared
    assert registration.modules == (shell.compute,)


def test_registration_discovers_units_once_and_deduplicates_managers():
    first = _Manager([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
    second = _Manager(manager_id=2)

    registration = register_flextensor_units(iter([first, first, second]))

    assert registration.modules == tuple(module.compute for _, module in (*first.units, *second.units))
    assert first.discovery_calls == second.discovery_calls == 1
    assert first.requests == second.requests == first.forward_calls == second.forward_calls == 0
    assert all(is_externally_managed_module(module) for module in registration.modules)
    assert all("forward" in module.__dict__ for module in registration.modules)
    assert "forward" not in first.model.__dict__
    assert all("forward" not in shell.__dict__ for _, shell in (*first.units, *second.units))


def test_registration_subset_retains_requested_order_and_identity():
    modules = [torch.nn.Linear(2, 2) for _ in range(3)]
    manager = _Manager(modules)
    shells = [shell for _, shell in manager.units]

    registration = register_flextensor_units([manager], modules=iter([shells[2], shells[0], shells[2]]))

    assert registration.modules == (modules[2], modules[0])
    assert not is_externally_managed_module(modules[1])
    assert "forward" not in modules[1].__dict__


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("phase", SimpleNamespace(value="profiling"), "reach inference"),
        ("compiled_offload_active", False, "compiled_offload_active=True"),
        ("compiled_replan_active", False, "exactly one active"),
        ("compiled_profile_active", True, "exactly one active"),
        ("compiled_replan_active", 1, "exactly one active"),
        ("offload_units", None, "public offload_units"),
        ("request_strategy_replan", None, "public request_strategy_replan"),
        ("model", None, "current model as a torch.nn.Module"),
        ("compiled_offload_manager_id", True, "integer compiled_offload_manager_id"),
        ("strategy_replan_status", SimpleNamespace(state="unknown", can_request=True, applied=None), "compatible"),
        ("strategy_replan_status", SimpleNamespace(state="idle", can_request=1, applied=None), "compatible"),
        ("strategy_replan_status", SimpleNamespace(state="idle", can_request=True, applied=1), "compatible"),
        (
            "strategy_replan_status",
            SimpleNamespace(state="in_progress", can_request=False, applied=None),
            "unavailable",
        ),
    ],
)
def test_manager_preflight_rejects_incompatible_provider_before_installation(attribute, value, message):
    first, second = _Manager(), _Manager(manager_id=2)
    second_units = [module.compute for _, module in second.units]
    setattr(second, attribute, value)

    with pytest.raises(RuntimeError, match=message):
        register_flextensor_units([first, second])

    modules = [module.compute for _, module in first.units] + second_units
    assert all("forward" not in module.__dict__ for module in modules)
    assert not any(is_externally_managed_module(module) for module in modules)


@pytest.mark.parametrize("ambiguity", ["manager_id", "shared_module", "duplicate_name", "duplicate_module"])
def test_registration_rejects_ambiguous_ownership_before_installation(ambiguity):
    first, second = _Manager(), _Manager(manager_id=2)
    module = first.units[0][1]
    if ambiguity == "manager_id":
        second.compiled_offload_manager_id = first.compiled_offload_manager_id
    elif ambiguity == "shared_module":
        second.model = torch.nn.Sequential(module)
        second.units = (("shared", module),)
    elif ambiguity == "duplicate_name":
        second.units = (("same", second.units[0][1]), ("same", torch.nn.Linear(2, 2)))
    else:
        second.units = (("first", second.units[0][1]), ("second", second.units[0][1]))

    with pytest.raises((RuntimeError, ValueError), match="share|unique"):
        register_flextensor_units([first, second])

    assert "forward" not in module.__dict__
    assert "forward" not in module.compute.__dict__
    assert not is_externally_managed_module(module.compute)


@pytest.mark.parametrize("targets", ["outside", "empty", "invalid_type"])
def test_subset_validation_precedes_installation(targets):
    manager = _Manager()
    selected = {"outside": [torch.nn.Linear(2, 2)], "empty": [], "invalid_type": [None]}[targets]

    with pytest.raises((TypeError, ValueError)):
        register_flextensor_units([manager], modules=selected)

    assert "forward" not in manager.units[0][1].compute.__dict__
    assert not is_externally_managed_module(manager.units[0][1].compute)


@pytest.mark.parametrize("stale_unit", [True, False])
def test_offload_units_must_belong_to_the_current_manager_model(stale_unit):
    first, second = _Manager(), _Manager(manager_id=2)
    current = second.units[0][1]
    invalid = torch.nn.Linear(2, 2) if stale_unit else object()
    second.units = (("invalid", invalid),)

    with pytest.raises(RuntimeError, match="belong to the manager's current model"):
        register_flextensor_units([first, second])

    for module in (first.units[0][1], current, invalid) if stale_unit else (first.units[0][1], current):
        assert "forward" not in module.__dict__
        assert not is_externally_managed_module(module)


@pytest.mark.parametrize("managers", [[], iter(())])
def test_empty_manager_batch_is_rejected(managers):
    with pytest.raises(ValueError, match="at least one"):
        register_flextensor_units(managers)


def test_existing_core_target_preflight_rejects_overlapping_units():
    child = torch.nn.Linear(2, 2)
    parent = _ParentModule(child)
    manager = _Manager([parent, child])

    with pytest.raises(ValueError, match="disjoint"):
        register_flextensor_units([manager])

    assert "forward" not in parent.__dict__
    assert "forward" not in child.__dict__
    assert not is_externally_managed_module(parent)


def test_existing_core_candidate_preflight_rejects_unqualified_backend():
    manager = _Manager()
    jit_config.strategy = OneBackendStrategy(SleepBackend())

    with pytest.raises(ValueError, match="External device management"):
        register_flextensor_units([manager])

    assert "forward" not in manager.units[0][1].compute.__dict__
    assert not is_externally_managed_module(manager.units[0][1].compute)


@pytest.mark.parametrize("recorded", [False, True])
def test_replan_rejects_pending_targets_without_requesting_or_running_forwards(recorded):
    manager = _Manager()
    registration = register_flextensor_units([manager])
    if recorded:
        manager.model(torch.ones(1, 2))

    with pytest.raises(RuntimeError, match="finish JIT tuning"):
        replan_flextensor(manager, registration, manager.forward)

    assert manager.requests == manager.forward_calls == 0


@pytest.mark.parametrize("prepared_forward", ["bound_method", "instance_function"])
def test_real_tuned_eager_candidate_replans_exact_requested_count_and_preserves_forward(prepared_forward):
    manager = _Manager()
    shell = manager.units[0][1]
    if prepared_forward == "instance_function":
        original = shell.forward
        shell.forward = lambda value: original(value)
    prepared = shell.forward
    registration = register_flextensor_units([manager])
    _settle(manager, registration)
    patched = registration._patched_modules[0]
    assert isinstance(patched._wrapper, TunedModule)
    signature = _tuned_forward_signature(patched)
    installed_forward = registration.modules[0].forward

    with torch.inference_mode():
        replan_flextensor(manager, registration, manager.forward)

    assert manager.requests == 1
    assert manager.forward_calls == manager.requested_count
    assert registration.modules[0].forward is not installed_forward
    assert all(
        actual is expected for actual, expected in zip(_tuned_forward_signature(patched), signature, strict=True)
    )
    assert manager.strategy_replan_status.applied is True
    assert shell.forward == prepared


@pytest.mark.parametrize("outcome", ["skipped", "dry_run", "fallback"])
def test_non_tuned_results_do_not_request_replanning(outcome, monkeypatch):
    manager = _Manager()
    if outcome == "skipped":
        jit_config.min_parameters = 100
    elif outcome == "dry_run":
        jit_config.dry_run = True
        jit_config.dry_run_failure_probability = 0
    else:

        def fail_tuning(*args, **kwargs):
            raise RuntimeError("candidate cannot build")

        monkeypatch.setattr(PatchedModule, "_tune", fail_tuning)
    registration = register_flextensor_units([manager])
    with torch.inference_mode():
        manager.model(torch.ones(1, 2))
        jit_deferred()
        manager.model(torch.ones(1, 2))

        replan_flextensor(manager, registration, manager.forward)

    assert manager.requests == manager.forward_calls == 0
    expected_state = {"skipped": "skipped", "dry_run": "tuned", "fallback": "eager"}[outcome]
    assert registration.state_counts == {expected_state: 1}


def test_manager_without_selected_units_does_not_request_replanning():
    first, second = _Manager(), _Manager(manager_id=2)
    registration = register_flextensor_units([first, second], modules=[first.units[0][1]])

    replan_flextensor(second, registration, second.forward)

    assert second.requests == second.forward_calls == 0


def test_each_manager_requires_only_its_own_selected_targets_to_settle():
    first, second = _Manager(), _Manager(manager_id=2)
    registration = register_flextensor_units([first, second])
    with torch.inference_mode():
        first.model(torch.ones(1, 2))
        jit_deferred()
        first.model(torch.ones(1, 2))
        assert registration.state_counts == {"tuned": 1, "init": 1}

        replan_flextensor(first, registration, first.forward)

    assert first.forward_calls == first.requested_count
    with pytest.raises(RuntimeError, match="finish JIT tuning"):
        replan_flextensor(second, registration, second.forward)
    assert second.requests == second.forward_calls == 0


@pytest.mark.parametrize("count", [0, -1, True, 1.5, None])
def test_replan_rejects_invalid_requested_forward_count(count):
    manager = _Manager()
    registration = register_flextensor_units([manager])
    _settle(manager, registration)
    manager.requested_count = count

    with pytest.raises(RuntimeError, match="positive integer"):
        replan_flextensor(manager, registration, manager.forward)

    assert manager.requests == 1
    assert manager.forward_calls == 0


def test_replan_rechecks_provider_availability_before_request():
    manager = _Manager()
    registration = register_flextensor_units([manager])
    _settle(manager, registration)
    manager.strategy_replan_status.can_request = False

    with pytest.raises(RuntimeError, match="unavailable"):
        replan_flextensor(manager, registration, manager.forward)

    assert manager.requests == manager.forward_calls == 0


def _replace_boundary(manager, replacement):
    shell = manager.units[0][1]
    if replacement == "manager_id":
        manager.compiled_offload_manager_id += 1
    elif replacement == "model":
        manager.model = torch.nn.Sequential(shell)
    elif replacement == "model_forward":
        original = manager.model.forward
        manager.model.forward = lambda value: original(value)
    elif replacement == "shell":
        replacement_shell = FlexTensorUnit(shell.compute)
        manager.model[0] = replacement_shell
        manager.units = (("replacement", replacement_shell),)
    elif replacement == "compute":
        shell.compute = torch.nn.Linear(2, 2)
    else:
        original = shell.forward
        shell.forward = lambda value: original(value)


@pytest.mark.parametrize("replacement", ["manager_id", "model", "model_forward", "shell", "compute", "shell_forward"])
@pytest.mark.parametrize("timing", ["before", "during"])
def test_replan_rejects_replacement_of_public_residency_boundaries(replacement, timing):
    manager = _Manager()
    registration = register_flextensor_units([manager])
    _settle(manager, registration)
    if timing == "before":
        _replace_boundary(manager, replacement)

    def run_forward():
        result = manager.forward()
        if manager.forward_calls == manager.requested_count:
            _replace_boundary(manager, replacement)
        return result

    with pytest.raises(RuntimeError, match="replaced"):
        replan_flextensor(manager, registration, run_forward)

    expected_count = manager.requested_count if timing == "during" else 0
    assert manager.forward_calls == expected_count
    assert manager.requests == (timing == "during")


def test_replaced_compute_cannot_look_like_a_manager_with_no_selected_units():
    manager = _Manager()
    registration = register_flextensor_units([manager])
    manager.units[0][1].compute = torch.nn.Linear(2, 2)

    with pytest.raises(RuntimeError, match="replaced a residency shell"):
        replan_flextensor(manager, registration, manager.forward)

    assert manager.requests == manager.forward_calls == 0


def test_replan_rejects_an_unregistered_manager():
    first, other = _Manager(), _Manager(manager_id=2)
    registration = register_flextensor_units([first])

    with pytest.raises(ValueError, match="one of the managers supplied"):
        replan_flextensor(other, registration, other.forward)

    assert other.requests == other.forward_calls == 0


def test_replan_requires_a_callable_whole_model_forward():
    manager = _Manager()
    registration = register_flextensor_units([manager])
    _settle(manager, registration)

    with pytest.raises(TypeError, match="one complete manager model forward"):
        replan_flextensor(manager, registration, None)

    assert manager.requests == manager.forward_calls == 0


@pytest.mark.parametrize("failure", ["incomplete", "unapplied", "failed"])
def test_replan_requires_completed_applied_provider_status(failure):
    manager = _Manager()
    registration = register_flextensor_units([manager])
    _settle(manager, registration)
    manager.apply = failure != "unapplied"
    callbacks = []

    def run_forward():
        callbacks.append(None)
        if failure == "incomplete":
            with torch.inference_mode():
                return manager.model(torch.ones(1, 2))
        value = manager.forward()
        if failure == "failed":
            manager.strategy_replan_status = SimpleNamespace(state="failed", can_request=False, applied=False)
        return value

    with pytest.raises(RuntimeError, match="complete and apply"):
        replan_flextensor(manager, registration, run_forward)

    assert manager.requests == 1
    assert len(callbacks) == manager.requested_count


@pytest.mark.parametrize("origin", ["provider", "callback"])
def test_replan_propagates_original_provider_and_callback_exceptions(origin, monkeypatch):
    manager = _Manager()
    registration = register_flextensor_units([manager])
    _settle(manager, registration)
    expected = ValueError("original provider or callback failure")

    def fail():
        raise expected

    if origin == "provider":
        monkeypatch.setattr(manager, "request_strategy_replan", fail)
    with pytest.raises(ValueError) as actual:
        replan_flextensor(manager, registration, fail if origin == "callback" else manager.forward)

    assert actual.value is expected
    assert manager.forward_calls == 0


def _replace_dispatch(patched, replacement):
    module = patched.__wrapped__
    if replacement == "native_forward":
        module.forward = patched._original_forward
    elif replacement == "foreign_router":

        def router(wrapped, instance, args, kwargs):
            return wrapped(*args, **kwargs)

        module.forward = wrapt.decorator(router)(patched._original_forward)
    elif replacement == "foreign_owner":
        other = PatchedModule(torch.nn.Linear(2, 2))
        module.forward = wrapt.decorator(other._forward_router)(patched._original_forward)
    elif replacement == "different_bound_self":
        other = torch.nn.Linear(2, 2)
        module.forward = wrapt.decorator(patched._forward_router)(other.forward)
    elif replacement == "changed_original":
        patched._original_forward = lambda value: value
        patched._proxy_forward()
    elif replacement == "dispatcher":
        patched._wrapper = object()
    else:

        def forward(value):
            return value

        forward.__wrapped__ = patched._original_forward
        forward._self_wrapper = patched._forward_router
        module.forward = forward


@pytest.mark.parametrize(
    "replacement",
    [
        "native_forward",
        "foreign_router",
        "foreign_owner",
        "different_bound_self",
        "changed_original",
        "dispatcher",
        "forged",
    ],
)
def test_replan_detects_replacement_of_installed_aitune_dispatch(replacement):
    manager = _Manager()
    registration = register_flextensor_units([manager])
    _settle(manager, registration)
    patched = registration._patched_modules[0]
    original, dispatcher = patched._original_forward, patched._wrapper

    def run_forward():
        value = manager.forward()
        if manager.forward_calls == manager.requested_count:
            _replace_dispatch(patched, replacement)
        return value

    try:
        with pytest.raises(RuntimeError, match="replaced an installed AITune forward"):
            replan_flextensor(manager, registration, run_forward)
    finally:
        patched._original_forward, patched._wrapper = original, dispatcher

    assert manager.forward_calls == manager.requested_count


@pytest.mark.parametrize(
    "replacement", ["native_forward", "foreign_router", "foreign_owner", "different_bound_self", "forged"]
)
def test_preexisting_forward_replacement_fails_before_requesting_replan(replacement):
    manager = _Manager()
    registration = register_flextensor_units([manager])
    _settle(manager, registration)
    _replace_dispatch(registration._patched_modules[0], replacement)

    with pytest.raises(RuntimeError, match="replaced an installed AITune forward"):
        replan_flextensor(manager, registration, manager.forward)

    assert manager.requests == manager.forward_calls == 0
