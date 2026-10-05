# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cached JIT forward references remain callable after permanent unpatching."""

from unittest.mock import Mock

import pytest
import torch

from aitune.torch import jit_reset, register_for_jit_tuning
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.config import config as jit_config
from aitune.torch.jit.patched_module import ModuleState
from aitune.torch.jit.patcher import Patcher


@pytest.fixture(autouse=True)
def deferred_cpu_session():
    jit_config.mode = JITMode.TUNE_DEFERRED
    jit_config.device = torch.device("cpu")
    jit_config.batch_axis_required = False
    jit_config.min_parameters = -1


@pytest.mark.parametrize("state", [ModuleState.INIT, ModuleState.RECORDING, ModuleState.TUNED])
@pytest.mark.parametrize("switch_mode", [False, True])
def test_cached_forward_calls_original_after_reset(state, switch_mode):
    module = torch.nn.Linear(2, 2)
    value = torch.ones(1, 2)
    expected = module(value)
    if state == ModuleState.TUNED:
        strategy = Mock()
        strategy.clone.return_value = strategy
        strategy.backend_results = []
        jit_config.strategy = strategy
        jit_config.dry_run = True
        jit_config.dry_run_failure_probability = 0.0
    registration = register_for_jit_tuning([module])
    cached_forward = module.forward
    wrapper = Patcher._patched_modules[0]
    if state != ModuleState.INIT:
        module(value)
    if state == ModuleState.TUNED:
        Patcher.enable_tune_deferred()
        module(value)
    assert wrapper._state == state

    jit_reset()
    if switch_mode:
        jit_config.mode = JITMode.TUNE_EAGER
    call_count = registration.reports[0].call_count

    torch.testing.assert_close(cached_forward(input=value), expected)

    assert module.forward is wrapper._original_forward
    assert Patcher._patched_modules == []
    assert registration.state_counts == {"detached": 1}
    assert registration.reports[0].call_count == call_count


@pytest.mark.parametrize("temporarily_restored", [False, True])
def test_active_cached_forward_still_requires_reset_before_mode_change(temporarily_restored):
    module = torch.nn.Linear(2, 2)
    register_for_jit_tuning([module])
    cached_forward = module.forward
    wrapper = Patcher._patched_modules[0]
    if temporarily_restored:
        wrapper._restore_original_forward()
    jit_config.mode = JITMode.TUNE_EAGER

    with pytest.raises(RuntimeError, match=r"Call jit_reset\(\) before switching JIT modes"):
        cached_forward(torch.ones(1, 2))


def test_cached_forward_is_inert_after_registration_rollback(monkeypatch):
    modules = [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]
    value = torch.ones(1, 2)
    expected = modules[0](value)
    original_register = Patcher._register_module
    cached_forwards = []
    wrappers = []

    def register_module(module, **kwargs):
        if wrappers:
            raise RuntimeError("registration failed")
        wrapper = original_register(module, **kwargs)
        wrappers.append(wrapper)
        cached_forwards.append(module.forward)
        return wrapper

    monkeypatch.setattr(Patcher, "_register_module", register_module)

    with pytest.raises(RuntimeError, match="registration failed"):
        register_for_jit_tuning(modules)
    jit_config.mode = JITMode.TUNE_EAGER

    torch.testing.assert_close(cached_forwards[0](input=value), expected)

    assert Patcher._patched_modules == []
    assert Patcher._session_mode is None
    assert wrappers[0]._state == ModuleState.DETACHED
    assert wrappers[0]._call_count == 0
    assert modules[0].forward is wrappers[0]._original_forward


@pytest.mark.parametrize("state", [ModuleState.INIT, ModuleState.EAGER])
def test_cached_forward_stays_eager_after_permanent_unpatch(state):
    module = torch.nn.Linear(2, 2)
    value = torch.ones(1, 2)
    expected = module(value)
    register_for_jit_tuning([module])
    cached_forward = module.forward
    wrapper = Patcher._patched_modules[0]
    wrapper._update_state(state)

    wrapper._unpatch()
    jit_config.mode = JITMode.TUNE_EAGER

    torch.testing.assert_close(cached_forward(input=value), expected)

    assert wrapper._state == state
    assert wrapper._call_count == 0
    assert Patcher._patched_modules == []
    assert module.forward is wrapper._original_forward


def test_cached_forward_stays_callable_after_parameter_filter_skips_module():
    module = torch.nn.Linear(2, 2)
    value = torch.ones(1, 2)
    expected = module(value)
    jit_config.min_parameters = 100
    registration = register_for_jit_tuning([module])
    cached_forward = module.forward
    module(value)
    jit_config.mode = JITMode.TUNE_EAGER

    torch.testing.assert_close(cached_forward(input=value), expected)

    assert registration.state_counts == {"skipped": 1}
    assert registration.reports[0].call_count == 1
    assert Patcher._patched_modules == []


def test_failed_device_restoration_keeps_hooks_and_cached_forward_for_cleanup_retry(monkeypatch):
    module = torch.nn.Linear(2, 2)
    value = torch.ones(1, 2)
    expected = module(value)
    register_for_jit_tuning([module])
    cached_forward = module.forward
    wrapper = Patcher._patched_modules[0]
    pre_hook = Mock(return_value=None)
    hook = Mock(return_value=None)
    module.register_forward_pre_hook(pre_hook)
    module.register_forward_hook(hook)
    original_pre_hooks = dict(module._forward_pre_hooks)
    original_hooks = dict(module._forward_hooks)
    monkeypatch.setattr(
        wrapper,
        "_restore_device_attribute",
        Mock(side_effect=[RuntimeError("device restoration failed"), None]),
    )

    with pytest.raises(RuntimeError, match="device restoration failed"):
        wrapper._unpatch()

    assert Patcher._patched_modules == [wrapper]
    assert dict(module._forward_pre_hooks) == original_pre_hooks
    assert dict(module._forward_hooks) == original_hooks
    jit_config.mode = JITMode.TUNE_EAGER
    torch.testing.assert_close(cached_forward(input=value), expected)
    torch.testing.assert_close(module(value), expected)
    assert pre_hook.call_count == hook.call_count == 1

    wrapper._unpatch()

    torch.testing.assert_close(module(value), expected)
    assert pre_hook.call_count == hook.call_count == 2
    assert dict(module._forward_pre_hooks) == original_pre_hooks
    assert dict(module._forward_hooks) == original_hooks
    assert Patcher._patched_modules == []


def test_detached_cached_forward_is_callable_when_reset_cleanup_needs_retry(monkeypatch):
    module = torch.nn.Linear(2, 2)
    value = torch.ones(1, 2)
    expected = module(value)
    registration = register_for_jit_tuning([module])
    cached_forward = module.forward
    wrapper = Patcher._patched_modules[0]

    with monkeypatch.context() as failure:
        failure.setattr(wrapper, "_unpatch", Mock(side_effect=RuntimeError("unpatch failed")))
        with pytest.raises(RuntimeError, match="unpatch failed"):
            jit_reset()

    assert registration.state_counts == {"detached": 1}
    assert Patcher._patched_modules == [wrapper]
    jit_config.mode = JITMode.TUNE_EAGER
    torch.testing.assert_close(cached_forward(input=value), expected)

    jit_reset()

    assert module.forward is wrapper._original_forward
    assert Patcher._patched_modules == []
