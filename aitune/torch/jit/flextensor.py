# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit registration and synchronous replanning through FlexTensor's public API."""

from collections.abc import Callable, Iterable
from typing import Any

import torch

from aitune.torch.jit.registration import JITRegistration, _collect_modules, register_for_jit_tuning
from aitune.torch.module.tuned_module import TunedModule


def _status(manager: Any) -> Any:
    """Read a compatible, passive public strategy-replan snapshot."""
    status = getattr(manager, "strategy_replan_status", None)
    applied = getattr(status, "applied", object())
    if (
        getattr(status, "state", None) not in ("idle", "in_progress", "done", "failed")
        or not isinstance(getattr(status, "can_request", None), bool)
        or (applied is not None and not isinstance(applied, bool))
    ):
        raise RuntimeError("FlexTensor must expose a compatible public strategy_replan_status")
    return status


def _manager_units(manager: Any) -> tuple[torch.nn.Module, ...]:
    """Resolve stable, uniquely named units using only public manager APIs."""
    if getattr(getattr(manager, "phase", None), "value", None) != "inference":
        raise RuntimeError("FlexTensor must finish discovery and profiling and reach inference before registration")
    if getattr(manager, "external_compile_active", None) is not True:
        raise RuntimeError("FlexTensor must be prepared with external_compile=True and no compile_fn")
    if not callable(getattr(manager, "request_strategy_replan", None)):
        raise RuntimeError("FlexTensor must expose public request_strategy_replan()")
    if not callable(getattr(manager, "offload_units", None)):
        raise RuntimeError("FlexTensor must expose public offload_units()")
    _status(manager)
    model = getattr(manager, "model", None)
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError("FlexTensor must expose its current model as a torch.nn.Module")
    owned = {id(module) for module in model.modules()}
    modules: list[torch.nn.Module] = []
    names: set[str] = set()
    identities: set[int] = set()
    for unit in manager.offload_units():
        try:
            name, module = unit
        except (TypeError, ValueError) as error:
            raise RuntimeError("FlexTensor offload_units() must return (name, module) pairs") from error
        if not isinstance(name, str) or not name or name in names or id(module) in identities:
            raise RuntimeError("FlexTensor offload units must have unique names and module identities")
        if not isinstance(module, torch.nn.Module) or id(module) not in owned:
            raise RuntimeError("FlexTensor offload units must belong to the manager's current model")
        names.add(name)
        identities.add(id(module))
        modules.append(module)
    return tuple(modules)


def _manager_id(manager: Any) -> int:
    """Read the public manager identity used to reject ambiguous ownership."""
    manager_id = getattr(manager, "compiled_offload_manager_id", None)
    if not isinstance(manager_id, int) or isinstance(manager_id, bool):
        raise RuntimeError("FlexTensor must expose an integer compiled_offload_manager_id")
    return manager_id


def register_flextensor_units(
    managers: Iterable[Any], *, modules: Iterable[torch.nn.Module] | None = None
) -> JITRegistration:
    """Register all public offload units, or an explicitly selected subset, once per worker.

    Prepare every manager for external compilation and finish its discovery/profiling
    before calling this helper. Identity-deduplicated managers and their units retain
    discovery order. All managers must support strategy replanning before AITune installs
    wrappers. Source placement belongs to FlexTensor; AITune's existing external-placement
    candidate and target validation applies unchanged.

    The caller records representative whole-model forwards, arms deferred tuning when
    needed, and warms every selected route before calling :func:`replan_flextensor`.
    FlexTensor is optional and is not imported by this module.
    """
    unique_managers = tuple({id(manager): manager for manager in managers}.values())
    if not unique_managers:
        raise ValueError("managers must contain at least one FlexTensor manager")
    manager_ids: set[int] = set()
    discovered: dict[int, torch.nn.Module] = {}
    for manager in unique_managers:
        manager_id = _manager_id(manager)
        if manager_id in manager_ids:
            raise ValueError("Distinct FlexTensor managers cannot share a compiled-offload manager ID")
        manager_ids.add(manager_id)
        units = _manager_units(manager)
        if not _status(manager).can_request:
            raise RuntimeError("FlexTensor strategy replanning is unavailable; use a fresh worker")
        for module in units:
            if id(module) in discovered:
                raise ValueError("Distinct FlexTensor managers cannot share an offload unit")
            discovered[id(module)] = module
    targets = list(discovered.values()) if modules is None else _collect_modules(modules)
    if any(id(module) not in discovered for module in targets):
        raise ValueError("Every selected module must be returned by a supplied manager's offload_units()")
    return register_for_jit_tuning(targets, device_management="external")


def replan_flextensor(manager: Any, registration: JITRegistration, run_forward: Callable[[], Any]) -> None:
    """Measure and apply one manager's strategy after the selected AITune targets settle.

    Each ``run_forward()`` invocation must execute exactly one complete forward of this
    manager's model with representative inputs. The callback is invoked exactly as many
    times as FlexTensor requests; there is no hidden tuning or readiness loop. The caller
    must warm every route first, including lazy compilation, so measurements use stable
    tuned forwards. A selected eager backend is a real tuned selection; dry-run simulations,
    skipped targets and native eager fallbacks do not request replanning on their own.

    Successful completion requires FlexTensor to report ``state='done', applied=True``
    and preserve AITune's installed forwards. Provider and callback exceptions propagate.
    A failed or incomplete replan requires a fresh worker; this helper provides no rollback,
    recovery or reset API.
    """
    _manager_id(manager)
    unit_ids = {id(module) for module in _manager_units(manager)}
    selected = [patched for patched in registration._patched_modules if id(patched.__wrapped__) in unit_ids]
    if any(patched._state.value not in ("tuned", "eager", "skipped") for patched in selected):
        raise RuntimeError("Selected FlexTensor units must finish JIT tuning before strategy replanning")
    tuned = [
        patched for patched in selected if patched._state.value == "tuned" and isinstance(patched._wrapper, TunedModule)
    ]
    if not tuned:
        return
    if not callable(run_forward):
        raise TypeError("run_forward must be a callback executing one complete manager model forward")
    if not _status(manager).can_request:
        raise RuntimeError("FlexTensor strategy replanning is unavailable; use a fresh worker")
    forwards = [(patched.__wrapped__, patched.__wrapped__.forward) for patched in tuned]
    count = manager.request_strategy_replan()
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise RuntimeError("FlexTensor request_strategy_replan() must return a positive integer forward count")
    for _ in range(count):
        run_forward()
    status = _status(manager)
    if status.state != "done" or status.applied is not True:
        raise RuntimeError("FlexTensor did not complete and apply strategy replanning; use a fresh worker")
    if any(module.forward is not forward for module, forward in forwards):
        raise RuntimeError("FlexTensor strategy replanning replaced an installed AITune forward; use a fresh worker")
