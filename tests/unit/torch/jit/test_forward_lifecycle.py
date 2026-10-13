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


def test_tuned_automatic_parent_detaches_child_without_losing_prepared_hooks():
    trace = []

    class Child(torch.nn.Linear):
        def forward(self, value):
            trace.append("prepared")
            return super().forward(value) + 7

    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = Child(2, 2)

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
