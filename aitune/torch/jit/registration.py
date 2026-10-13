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
    unique: dict[int, torch.nn.Module] = {}
    for index, module in enumerate(materialized):
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"modules[{index}] must be a torch.nn.Module, got {type(module).__name__}")
        unique.setdefault(id(module), module)
    return list(unique.values())


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
    modules: Iterable[torch.nn.Module], *, device_management: Literal["aitune", "external"] = "aitune"
) -> JITRegistration:
    """Register pre-existing modules as independent JIT tuning targets.

    Unlike automatic JIT interception, registration does not patch construction of future
    ``torch.nn.Module`` instances. Each supplied module is treated as an independent tuning
    head and otherwise follows the active JIT configuration. Supply the complete target set
    once in an unused worker. Automatic discovery and explicit registration are alternative
    entry paths, and the tuning mode stays fixed after the first wrapper is installed.

    Validation runs before installation. A setup failure after installation begins may leave
    earlier targets patched; it propagates without rollback and requires a fresh worker.

    Args:
        modules: Iterable of existing modules. Pass ``[module]`` for one module. Passing an
            ``nn.ModuleList`` directly registers its elements.
        device_management: ``"aitune"`` permits ordinary source placement. ``"external"``
            preserves the fixed source tree and prepared forwarding, without overriding its
            device attribute. Configure only eager or qualified Inductor JIT candidates first.
            Independent internal copies and recorded inputs may still move. The caller must
            avoid shared tensors/storage with unrelated unprotected trees.

    Returns:
        A read-only registration view with live module states and inspection reports.

    Raises:
        TypeError: If ``modules`` is not iterable or contains a non-module value.
        ValueError: If ``modules`` is empty, a target is excluded, target ownership trees overlap,
            or the device policy/candidate configuration is unsupported.
        RuntimeError: If inspection mode or either JIT entry path is already active.
    """
    if device_management not in ("aitune", "external"):
        raise ValueError("device_management must be 'aitune' or 'external'")
    return JITRegistration(Patcher.register_modules(_materialize_modules(modules), device_management=device_management))
