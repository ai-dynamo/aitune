# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cached JIT forward references remain callable through ordinary skip and eager fallback."""

import pytest
import torch

from aitune.torch import register_for_jit_tuning
from aitune.torch.backend import TorchEagerBackend
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.config import config as jit_config
from aitune.torch.jit.patched_module import ModuleState
from aitune.torch.jit.patcher import Patcher, prepare_for_jit_tuning
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy
from tests.toy_backends import BuildFailsBackend


@pytest.fixture(autouse=True)
def deferred_cpu_session():
    jit_config.mode = JITMode.TUNE_DEFERRED
    jit_config.device = torch.device("cpu")
    jit_config.batch_axis_required = False
    jit_config.min_parameters = -1


@pytest.mark.parametrize("state", [ModuleState.EAGER, ModuleState.DETACHED])
def test_cached_forward_stays_eager_after_permanent_unpatch(state):
    module = torch.nn.Linear(2, 2)
    value = torch.ones(1, 2)
    expected = module(value)
    register_for_jit_tuning([module])
    cached_forward = module.forward
    wrapper = Patcher._patched_modules[0]
    wrapper._update_state(state)

    wrapper._unpatch()

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

    torch.testing.assert_close(cached_forward(input=value), expected)

    assert registration.state_counts == {"skipped": 1}
    assert registration.reports[0].call_count == 1
    assert Patcher._patched_modules == []


@pytest.mark.parametrize("failure", ["dry-run", "backend-build"])
def test_tuning_failure_preserves_prepared_forward_and_hooks(failure):
    jit_config.strategy = OneBackendStrategy(BuildFailsBackend(ValueError)).enable_performance_validation(False)
    jit_config.dry_run = failure == "dry-run"
    jit_config.dry_run_failure_probability = 1.0
    module = torch.nn.Linear(2, 2)
    original_forward = module.forward
    trace = []

    def prepared(value):
        trace.append("prepared")
        return original_forward(value) + 7

    module.forward = prepared
    module.register_forward_pre_hook(lambda *_: trace.append("pre-hook"))
    module.register_forward_hook(lambda *_: trace.append("post-hook"))
    value = torch.ones(1, 2)
    expected = module(value)
    registration = register_for_jit_tuning([module])
    module(value)
    Patcher.enable_tune_deferred()
    module(value)

    assert registration.state_counts == {"eager": 1}
    assert module.forward is prepared
    for _ in range(2):
        trace.clear()
        torch.testing.assert_close(module(value), expected)
        assert trace == ["pre-hook", "prepared", "post-hook"]


@pytest.mark.parametrize("owns_child", [True, False])
def test_tuned_automatic_parent_detaches_child_without_losing_prepared_hooks(owns_child):
    trace = []

    class Child(torch.nn.Linear):
        def forward(self, value):
            trace.append("prepared")
            return super().forward(value) + 7

    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            child = Child(2, 2)
            if owns_child:
                self.child = child
            else:
                object.__setattr__(self, "child", child)

        def forward(self, value):
            return self.child(value)

    jit_config.max_depth_level = 2
    jit_config.strategy = OneBackendStrategy(TorchEagerBackend()).enable_performance_validation(False)
    with prepare_for_jit_tuning():
        parent = Parent()
    parent.child.register_forward_pre_hook(lambda *_: trace.append("pre-hook"))
    parent.child.register_forward_hook(lambda *_: trace.append("post-hook"))
    child_wrapper = next(wrapper for wrapper in Patcher._patched_modules if wrapper.__wrapped__ is parent.child)
    parent_wrapper = next(wrapper for wrapper in Patcher._patched_modules if wrapper.__wrapped__ is parent)
    value = torch.ones(1, 2)
    expected = parent(value)
    Patcher.enable_tune_deferred()
    parent(value)

    assert parent_wrapper._state is ModuleState.TUNED
    assert "TorchEagerBackend" in parent_wrapper._extra_state_info
    assert child_wrapper._state is ModuleState.DETACHED
    for _ in range(2):
        trace.clear()
        torch.testing.assert_close(parent(value), expected)
        assert trace == ["pre-hook", "prepared", "post-hook"]


def test_tuning_an_explicit_caller_preserves_its_independently_registered_callee():
    class Caller(torch.nn.Module):
        def __init__(self, callee):
            super().__init__()
            object.__setattr__(self, "callee", callee)

        def forward(self, value):
            return self.callee(value)

    jit_config.detect_graph_breaks = False
    jit_config.strategy = OneBackendStrategy(TorchEagerBackend()).enable_performance_validation(False)
    callee = torch.nn.Linear(2, 2)
    caller = Caller(callee)
    value = torch.ones(1, 2)
    expected = callee(value)
    registration = register_for_jit_tuning([caller, callee])
    caller(value)
    Patcher.enable_tune_deferred()
    caller(value)

    assert registration.state_counts == {"tuned": 2}
    assert set(registration._patched_modules) == set(Patcher._patched_modules)
    torch.testing.assert_close(caller(value), expected)


def test_tuned_parent_restores_hooks_on_an_uncalled_owned_child():
    trace = []

    class Child(torch.nn.Linear):
        def forward(self, value):
            trace.append("prepared")
            return super().forward(value) + 7

    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.used = Child(2, 2)
            self.unused = Child(2, 2)

        def forward(self, value):
            return self.used(value)

    jit_config.max_depth_level = 2
    jit_config.detect_graph_breaks = False
    jit_config.strategy = OneBackendStrategy(TorchEagerBackend()).enable_performance_validation(False)
    with prepare_for_jit_tuning():
        parent = Parent()
    parent.unused.register_forward_pre_hook(lambda *_: trace.append("pre-hook"))
    parent.unused.register_forward_hook(lambda *_: trace.append("post-hook"))
    unused_wrapper = next(wrapper for wrapper in Patcher._patched_modules if wrapper.__wrapped__ is parent.unused)
    value = torch.ones(1, 2)
    expected = torch.nn.functional.linear(value, parent.unused.weight, parent.unused.bias) + 7
    parent(value)
    assert unused_wrapper._fq_name is None
    Patcher.enable_tune_deferred()
    parent(value)

    assert unused_wrapper._state is ModuleState.DETACHED
    assert unused_wrapper not in Patcher._patched_modules
    trace.clear()
    torch.testing.assert_close(parent.unused(value), expected)
    assert trace == ["pre-hook", "prepared", "post-hook"]


def test_parent_failure_resumes_uncalled_and_insufficiently_sampled_children():
    trace = []

    class Child(torch.nn.Linear):
        def forward(self, value):
            trace.append("prepared")
            return super().forward(value) + 7

    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = Child(2, 2)
            self.unused = Child(2, 2)

        def forward(self, value, use_child):
            return self.child(value) if use_child else value

    jit_config.mode = JITMode.TUNE_EAGER
    jit_config.min_samples = 3
    jit_config.max_depth_level = 2
    jit_config.detect_graph_breaks = False
    jit_config.strategy = OneBackendStrategy(BuildFailsBackend(ValueError)).enable_performance_validation(False)
    with prepare_for_jit_tuning():
        parent = Parent()
    for child in (parent.child, parent.unused):
        child.register_forward_pre_hook(lambda *_: trace.append("pre-hook"))
        child.register_forward_hook(lambda *_: trace.append("post-hook"))
    wrappers = {id(wrapper.__wrapped__): wrapper for wrapper in Patcher._patched_modules}
    parent_wrapper = wrappers[id(parent)]
    child_wrapper = wrappers[id(parent.child)]
    unused_wrapper = wrappers[id(parent.unused)]
    value = torch.ones(1, 2)
    parent(value, True)
    parent(value, False)
    parent(value, False)

    assert parent_wrapper._state is ModuleState.EAGER
    assert child_wrapper._state is ModuleState.RECORDING
    assert child_wrapper._call_count == 1
    assert unused_wrapper._state is ModuleState.INIT
    for child, wrapper, count in ((parent.child, child_wrapper, 2), (parent.unused, unused_wrapper, 1)):
        expected = torch.nn.functional.linear(value, child.weight, child.bias) + 7
        trace.clear()
        torch.testing.assert_close(child(value), expected)
        assert trace == ["pre-hook", "prepared", "post-hook"]
        assert wrapper._state is ModuleState.RECORDING
        assert wrapper._call_count == count


def test_child_hook_restoration_failure_propagates_after_successful_backend_build():
    failure = RuntimeError("child hook restoration failed")

    class Child(torch.nn.Linear):
        reject_hook_restore = False

        def __setattr__(self, name, value):
            if name == "_forward_hooks" and value and self.reject_hook_restore:
                self.reject_hook_restore = False
                raise failure
            super().__setattr__(name, value)

    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = Child(2, 2)

        def forward(self, value):
            return self.child(value)

    class HookRestoreFailsBackend(TorchEagerBackend):
        def _build(self, module, graph_spec, samples, cache_dir):
            built = super()._build(module, graph_spec, samples, cache_dir)
            assert isinstance(module, Parent)
            module.child.reject_hook_restore = True
            return built

    jit_config.max_depth_level = 2
    jit_config.detect_graph_breaks = False
    jit_config.strategy = OneBackendStrategy(HookRestoreFailsBackend()).enable_performance_validation(False)
    with prepare_for_jit_tuning():
        parent = Parent()
    parent.child.register_forward_hook(lambda *_: None)
    parent_wrapper = next(wrapper for wrapper in Patcher._patched_modules if wrapper.__wrapped__ is parent)
    value = torch.ones(1, 2)
    parent(value)
    Patcher.enable_tune_deferred()

    with pytest.raises(RuntimeError) as raised:
        parent(value)

    assert raised.value is failure
    assert parent_wrapper._state is ModuleState.TUNED
    assert "HookRestoreFailsBackend" in parent_wrapper._extra_state_info
