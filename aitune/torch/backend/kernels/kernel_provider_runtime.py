# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime activation of a selected kernel optimization plan."""

from collections import defaultdict
from collections.abc import Callable, Generator
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from aitune.torch.backend.kernels.kernel_optimization_plan import KernelOptimizationPlan
from aitune.torch.backend.kernels.kernel_provider import KernelProvider


class KernelProviderRuntime:
    """Install a kernel optimization plan by swapping ``torch.nn.functional`` calls."""

    def __init__(self, module: nn.Module, plan: KernelOptimizationPlan) -> None:
        """Initialize runtime state without running kernel optimization."""
        self.module = module
        self.plan = plan
        self._provider_functions = {
            id(provider): self._create_provider_function(provider) for provider in plan.providers
        }
        self._function_stacks: dict[str, list[Callable]] = defaultdict(list)
        self._hooks: list = []

    @property
    def is_active(self) -> bool:
        """Return whether runtime hooks are currently installed."""
        return bool(self._hooks)

    def activate(self) -> None:
        """Activate the selected providers idempotently."""
        if self._hooks:
            return

        hooks = []
        try:
            if self.plan.providers:
                hooks.append(self.module.register_forward_pre_hook(self._create_pre_hook()))
                hooks.append(self.module.register_forward_hook(self._create_post_hook(), always_call=True))
        except Exception:
            for hook in hooks:
                hook.remove()
            raise
        self._hooks = hooks

    def deactivate(self) -> None:
        """Deactivate providers idempotently."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    @contextmanager
    def applied(self) -> Generator[None, None, None]:
        """Temporarily activate the plan for inference while preserving prior runtime state."""
        was_active = self.is_active
        if not was_active:
            self.activate()
        try:
            with torch.no_grad():
                yield
        finally:
            if not was_active:
                self.deactivate()

    def _create_pre_hook(self):
        """Create a hook that installs providers before each module forward."""

        def pre_hook(_module, *_args, **_kwargs):
            """Install providers transactionally for the current forward call."""
            patched_function_names = []
            try:
                for provider in self.plan.providers:
                    function_name = provider.supported_function
                    self._function_stacks[function_name].append(getattr(F, function_name))
                    setattr(F, function_name, self._provider_functions[id(provider)])
                    patched_function_names.append(function_name)
            except Exception:
                for function_name in reversed(patched_function_names):
                    setattr(F, function_name, self._function_stacks[function_name].pop())
                raise

        return pre_hook

    def _create_post_hook(self):
        """Create a hook that restores functions after each module forward."""

        def post_hook(_module, *_args, **_kwargs):
            """Restore functions patched for the current forward call."""
            for provider in reversed(self.plan.providers):
                function_name = provider.supported_function
                if self._function_stacks[function_name]:
                    setattr(F, function_name, self._function_stacks[function_name].pop())

        return post_hook

    @staticmethod
    def _create_provider_function(provider: KernelProvider) -> Callable:
        """Expose a callable provider as a regular Python function."""

        def provider_function(*args, **kwargs):
            """Call the configured kernel provider."""
            return provider(*args, **kwargs)

        provider_function.__name__ = provider.supported_function
        provider_function.__qualname__ = provider.supported_function
        return provider_function
