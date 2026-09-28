# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit registration of existing modules for JIT tuning."""

from collections import Counter
from collections.abc import Iterable
from typing import Literal

import torch

from aitune.torch.jit.patched_module import PatchedModule
from aitune.torch.jit.patcher import Patcher
from aitune.torch.tune_data.report_models import ModuleInspectionReport

DeviceManagement = Literal["aitune", "external"]
_DEVICE_MANAGEMENT_POLICIES = ("aitune", "external")


def _materialize_modules(modules: Iterable[torch.nn.Module]) -> list[torch.nn.Module]:
    """Materialize and validate an explicit module iterable without mutating JIT state."""
    try:
        materialized = list(modules)
    except TypeError as error:
        raise TypeError(
            "modules must be an iterable of torch.nn.Module instances; use [module] for one module"
        ) from error

    if not materialized:
        raise ValueError("modules must contain at least one torch.nn.Module")
    for index, module in enumerate(materialized):
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"modules[{index}] must be a torch.nn.Module, got {type(module).__name__}")
    return materialized


def _unique_modules(modules: list[torch.nn.Module]) -> list[torch.nn.Module]:
    """Deduplicate targets by identity while retaining input order."""
    unique: list[torch.nn.Module] = []
    seen_module_ids: set[int] = set()
    for module in modules:
        if id(module) not in seen_module_ids:
            seen_module_ids.add(id(module))
            unique.append(module)
    return unique


class JITRegistration:
    """Read-only live view of modules explicitly registered for JIT tuning."""

    __slots__ = ("_patched_modules",)

    def __init__(self, patched_modules: tuple[PatchedModule, ...]) -> None:
        """Initialize a registration returned by :func:`register_for_jit_tuning`."""
        self._patched_modules = patched_modules

    @property
    def modules(self) -> tuple[torch.nn.Module, ...]:
        """Modules in deduplicated registration order."""
        return tuple(module.__wrapped__ for module in self._patched_modules)

    @property
    def reports(self) -> tuple[ModuleInspectionReport, ...]:
        """Fresh inspection snapshots for the registered modules."""
        return tuple(module.inspection_report() for module in self._patched_modules)

    @property
    def state_counts(self) -> dict[str, int]:
        """Number of registered modules in each current JIT state."""
        return dict(Counter(module._state.value for module in self._patched_modules))

    @property
    def all_tuned(self) -> bool:
        """Whether every registered module reached the JIT tuned state.

        This reports JIT state, not guaranteed acceleration: an eager backend or a
        successful dry-run simulation may also produce the tuned state.
        """
        return bool(self._patched_modules) and all(module._state.value == "tuned" for module in self._patched_modules)


def register_for_jit_tuning(
    modules: Iterable[torch.nn.Module],
    *,
    device_management: DeviceManagement = "aitune",
) -> JITRegistration:
    """Register pre-existing modules as independent JIT tuning targets.

    Unlike automatic JIT interception, registration does not patch construction of future
    ``torch.nn.Module`` instances. Each supplied module is treated as an independent tuning
    head and otherwise follows the active JIT configuration.

    Args:
        modules: Iterable of existing modules. Pass ``[module]`` for one module. Passing an
            ``nn.ModuleList`` directly registers its elements.
        device_management: ``"aitune"`` allows AITune to place the original module as usual.
            ``"external"`` preserves placement of the original module tree while allowing
            inputs, backend artifacts, and internal copies to move.

    Returns:
        A read-only registration view with live module states and inspection reports.

    Raises:
        TypeError: If ``modules`` is not iterable or contains a non-module value.
        ValueError: If ``modules`` is empty, the device policy is invalid, a target is excluded,
            or target ownership trees overlap.
        RuntimeError: If inspection mode or global constructor interception is active.
    """
    if device_management not in _DEVICE_MANAGEMENT_POLICIES:
        policies = ", ".join(repr(policy) for policy in _DEVICE_MANAGEMENT_POLICIES)
        raise ValueError(f"device_management must be one of: {policies}")

    Patcher.validate_explicit_registration_session()
    materialized_modules = _materialize_modules(modules)
    patched_modules = Patcher.register_modules(
        _unique_modules(materialized_modules),
        device_management=device_management,
    )
    return JITRegistration(patched_modules)
