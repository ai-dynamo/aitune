# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for runtime activation of kernel optimization plans."""

from collections.abc import Callable
from types import FunctionType

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from aitune.torch.backend.kernels.kernel_optimization_plan import KernelOptimizationPlan
from aitune.torch.backend.kernels.kernel_provider import KernelProvider, KernelProviderState
from aitune.torch.backend.kernels.kernel_provider_runtime import KernelProviderRuntime


class _CallableProvider(KernelProvider):
    def __init__(self, function_name: str, function: Callable):
        super().__init__()
        self._function_name = function_name
        self.function = function
        self.state = KernelProviderState.READY

    @property
    def supported_function(self) -> str:
        return self._function_name

    def _prepare(self, samples) -> bool:
        return True

    def _infer(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def _to_dict(self) -> dict:
        raise NotImplementedError

    @classmethod
    def _from_dict(cls, state_dict: dict) -> "_CallableProvider":
        raise NotImplementedError


class _ReluModule(nn.Module):
    def forward(self, value):
        return F.relu(value)


class _IntrospectingReluModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.installed_function = None

    def forward(self, value):
        self.installed_function = F.relu
        return F.relu(value)


def _runtime(module: nn.Module, provider: KernelProvider):
    plan = KernelOptimizationPlan((provider,))
    return KernelProviderRuntime(module, plan)


def test_runtime_activate_and_deactivate_are_idempotent():
    calls = []
    original_relu = F.relu
    module = _ReluModule()
    runtime = _runtime(
        module,
        _CallableProvider("relu", lambda value: calls.append(value) or original_relu(value)),
    )

    runtime.activate()
    runtime.activate()

    assert runtime.is_active
    assert len(module._forward_pre_hooks) == 1
    assert len(module._forward_hooks) == 1
    module(torch.tensor([-1.0, 1.0]))
    assert len(calls) == 1

    runtime.deactivate()
    runtime.deactivate()

    assert not runtime.is_active
    assert len(module._forward_pre_hooks) == 0
    assert len(module._forward_hooks) == 0
    module(torch.tensor([-1.0, 1.0]))
    assert len(calls) == 1


def test_runtime_activation_rolls_back_when_hook_registration_fails(monkeypatch):
    module = _ReluModule()
    runtime = _runtime(module, _CallableProvider("relu", F.relu))

    def failing_register_forward_hook(*_args, **_kwargs):
        raise RuntimeError("hook registration failed")

    monkeypatch.setattr(module, "register_forward_hook", failing_register_forward_hook)

    with pytest.raises(RuntimeError, match="hook registration failed"):
        runtime.activate()

    assert not runtime.is_active
    assert len(module._forward_pre_hooks) == 0
    assert len(module._forward_hooks) == 0


def test_runtime_installs_function_wrapper_around_provider():
    original_relu = F.relu
    provider = _CallableProvider("relu", original_relu)
    module = _IntrospectingReluModule()
    runtime = _runtime(module, provider)

    with runtime.applied():
        module(torch.tensor([-1.0, 1.0]))

    assert isinstance(module.installed_function, FunctionType)
    assert module.installed_function is not provider
    assert module.installed_function.__name__ == "relu"
    assert module.installed_function.__qualname__ == "relu"
    assert not hasattr(provider, "__name__")
    assert F.relu is original_relu


def test_plan_apply_temporarily_activates_providers():
    calls = []
    original_relu = F.relu
    module = _ReluModule()
    plan = KernelOptimizationPlan((
        _CallableProvider("relu", lambda value: calls.append(value) or original_relu(value)),
    ))

    with plan.apply(module):
        module(torch.tensor([-1.0, 1.0]))

    module(torch.tensor([-1.0, 1.0]))

    assert len(calls) == 1
    assert len(module._forward_pre_hooks) == 0
    assert len(module._forward_hooks) == 0
    assert F.relu is original_relu


def test_plan_apply_enters_no_grad_and_restores_previous_state():
    grad_enabled = []
    module = _ReluModule()
    plan = KernelOptimizationPlan((
        _CallableProvider(
            "relu",
            lambda value: grad_enabled.append(torch.is_grad_enabled()) or value,
        ),
    ))

    with torch.enable_grad():
        with plan.apply(module):
            module(torch.tensor([1.0]))
        assert torch.is_grad_enabled()

    with torch.no_grad():
        with plan.apply(module):
            module(torch.tensor([1.0]))
        assert not torch.is_grad_enabled()

    assert grad_enabled == [False, False]


def test_runtime_restores_function_when_forward_raises():
    original_relu = F.relu
    module = _ReluModule()

    def failing_provider(value):
        raise RuntimeError("provider failure")

    runtime = _runtime(module, _CallableProvider("relu", failing_provider))
    runtime.activate()

    with pytest.raises(RuntimeError, match="provider failure"):
        module(torch.tensor([-1.0, 1.0]))

    assert F.relu is original_relu
    assert runtime._function_stacks["relu"] == []
    runtime.deactivate()


def test_runtime_rolls_back_partial_patches_when_pre_hook_raises():
    original_relu = F.relu
    missing_function_name = "aitune_missing_function"
    module = _ReluModule()
    plan = KernelOptimizationPlan((
        _CallableProvider("relu", original_relu),
        _CallableProvider(missing_function_name, original_relu),
    ))
    runtime = KernelProviderRuntime(module, plan)
    runtime.activate()

    try:
        with pytest.raises(AttributeError, match=missing_function_name):
            module(torch.tensor([-1.0, 1.0]))

        assert F.relu is original_relu
        assert runtime._function_stacks["relu"] == []
        assert runtime._function_stacks[missing_function_name] == []
    finally:
        F.relu = original_relu
        runtime.deactivate()


def test_applied_preserves_previously_active_state():
    module = _ReluModule()
    runtime = _runtime(module, _CallableProvider("relu", F.relu))
    runtime.activate()

    with runtime.applied():
        assert runtime.is_active

    assert runtime.is_active
    runtime.deactivate()


def test_plan_apply_deactivates_runtime_when_context_raises():
    module = _ReluModule()
    plan = KernelOptimizationPlan((_CallableProvider("relu", F.relu),))

    with pytest.raises(RuntimeError, match="context failure"):
        with plan.apply(module):
            assert len(module._forward_pre_hooks) == 1
            assert len(module._forward_hooks) == 1
            raise RuntimeError("context failure")

    assert len(module._forward_pre_hooks) == 0
    assert len(module._forward_hooks) == 0
