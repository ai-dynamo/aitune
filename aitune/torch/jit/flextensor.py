# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit registration and synchronous replanning through FlexTensor's public API."""

from collections.abc import Callable, Iterable
from types import MethodType
from typing import Any, NamedTuple

import torch
import wrapt

from aitune.torch.jit.patched_module import PatchedModule
from aitune.torch.jit.registration import JITRegistration, _collect_modules, register_for_jit_tuning
from aitune.torch.module.tuned_module import TunedModule


class FlexTensorUnit(torch.nn.Module):
    """Keep FlexTensor's residency boundary outside the original compute module.

    Insert these shells into the model **before** calling ``flextensor.offload`` and
    select the shells with its include patterns, so FlexTensor skips their descendants.
    FlexTensor owns the shell's prepared forward; AITune registers only ``compute``.
    Thus ordinary AITune warmup, correctness checks and profiling can repeat compute
    while the enclosing FlexTensor call keeps its weights resident.

    The original module, parameters, hooks and outputs are retained. The extra child
    adds ``compute.`` to parameter and checkpoint paths; checkpoint keys are not remapped.
    """

    def __init__(self, compute: torch.nn.Module) -> None:
        """Wrap an existing compute module without copying or moving it."""
        super().__init__()
        if not isinstance(compute, torch.nn.Module):
            raise TypeError("compute must be a torch.nn.Module")
        if any(isinstance(module, FlexTensorUnit) for module in compute.modules()):
            raise ValueError("FlexTensorUnit shells cannot be nested or wrapped twice")
        self.compute = compute
        self.training = compute.training

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate arguments and return values unchanged to the original module."""
        return self.compute(*args, **kwargs)


class _ManagerBoundary(NamedTuple):
    """Original public ownership and forwards associated with one registration."""

    manager: Any
    manager_id: int
    model: torch.nn.Module
    forward: tuple[object, object]
    units: tuple[tuple[FlexTensorUnit, torch.nn.Module, tuple[object, object]], ...]


class _FlexTensorRegistration(JITRegistration):
    """Keep immutable public boundary associations alongside the ordinary JIT view."""

    __slots__ = ("_boundaries",)

    def __init__(self, registration: JITRegistration, boundaries: tuple[_ManagerBoundary, ...]) -> None:
        """Retain the unchanged core registration's targets and their public owners."""
        super().__init__(registration._patched_modules)
        self._boundaries = boundaries


def _replan_mode(manager: Any) -> bool:
    """Validate the active public compiled-offload mode without arming a tail."""
    if getattr(manager, "compiled_offload_active", None) is not True:
        raise RuntimeError("FlexTensor must have compiled_offload_active=True")
    replan = getattr(manager, "compiled_replan_active", None)
    profile = getattr(manager, "compiled_profile_active", None)
    if not isinstance(replan, bool) or not isinstance(profile, bool) or replan == profile:
        raise RuntimeError(
            "FlexTensor must expose exactly one active compiled_replan_active or compiled_profile_active mode"
        )
    if replan:
        if not callable(getattr(manager, "request_strategy_replan", None)):
            raise RuntimeError("FlexTensor must expose public request_strategy_replan()")
        _status(manager)
    return replan


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


def _residency_shell(module: torch.nn.Module, owned: set[int]) -> FlexTensorUnit:
    """Require an integration-prepared shell with its original live compute child."""
    if not isinstance(module, FlexTensorUnit):
        raise RuntimeError(
            "Unsafe FlexTensor unit replay: insert FlexTensorUnit(original) before flextensor.offload(), "
            "and select its shell rather than its compute descendants with include_patterns"
        )
    if not isinstance(module.compute, torch.nn.Module) or id(module.compute) not in owned:
        raise RuntimeError("FlexTensorUnit compute must belong to the manager's current model")
    if any(isinstance(child, FlexTensorUnit) for child in module.compute.modules()):
        raise RuntimeError("FlexTensorUnit must contain one original compute module, not nested shells")
    return module


def _manager_units(manager: Any) -> tuple[FlexTensorUnit, ...]:
    """Resolve stable, uniquely named units using only public manager APIs."""
    if getattr(getattr(manager, "phase", None), "value", None) != "inference":
        raise RuntimeError("FlexTensor must finish discovery and profiling and reach inference before registration")
    _replan_mode(manager)
    if not callable(getattr(manager, "offload_units", None)):
        raise RuntimeError("FlexTensor must expose public offload_units()")
    model = getattr(manager, "model", None)
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError("FlexTensor must expose its current model as a torch.nn.Module")
    owned = {id(module) for module in model.modules()}
    modules: list[FlexTensorUnit] = []
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
        modules.append(_residency_shell(module, owned))
    return tuple(modules)


def _manager_id(manager: Any) -> int:
    """Read the public manager identity used to reject ambiguous ownership."""
    manager_id = getattr(manager, "compiled_offload_manager_id", None)
    if not isinstance(manager_id, int) or isinstance(manager_id, bool):
        raise RuntimeError("FlexTensor must expose an integer compiled_offload_manager_id")
    return manager_id


def _tuned_forward_signature(patched: PatchedModule) -> tuple[object, object, object]:
    """Identify owned dispatch across proxies recreated by ordinary tuned inference."""
    forward = patched.__wrapped__.forward
    router = getattr(forward, "_self_wrapper", None)
    if (
        type(forward) is not wrapt.FunctionWrapper
        or getattr(router, "__self__", None) is not patched
        or getattr(router, "__func__", None) is not patched._forward_router.__func__
        or forward.__wrapped__ is not patched._original_forward
        or not isinstance(patched._wrapper, TunedModule)
    ):
        raise RuntimeError("FlexTensor strategy replanning replaced an installed AITune forward; use a fresh worker")
    return forward.__wrapped__, router.__func__, patched._wrapper


def register_flextensor_units(
    managers: Iterable[Any], *, modules: Iterable[torch.nn.Module] | None = None
) -> JITRegistration:
    """Register all public offload units, or an explicitly selected subset, once per worker.

    Insert :class:`FlexTensorUnit` shells before offloading and select only those shells
    in FlexTensor. Finish discovery/profiling before registration: the final live public
    ``offload_units()`` shells supply the inner ``compute`` targets. ``modules`` optionally
    selects public shells, not compute children. The returned registration exposes computes.
    Identity-deduplicated managers and their units retain discovery order. Source placement
    belongs to FlexTensor; AITune's existing external-placement validation applies unchanged.

    Compiled replan mode (including ``external_compile=True``) must have an available
    public replan request. ``compile_fn`` with retained view profiling is also supported;
    that mode keeps its measured strategy and must not call :func:`replan_flextensor`.

    The caller records representative whole-model forwards, arms deferred tuning when
    needed, and warms every selected route before calling :func:`replan_flextensor`.
    FlexTensor is optional and is not imported by this module.
    """
    unique_managers = tuple({id(manager): manager for manager in managers}.values())
    if not unique_managers:
        raise ValueError("managers must contain at least one FlexTensor manager")
    manager_ids: set[int] = set()
    discovered: dict[int, FlexTensorUnit] = {}
    computes: set[int] = set()
    boundaries: list[_ManagerBoundary] = []
    for manager in unique_managers:
        manager_id = _manager_id(manager)
        if manager_id in manager_ids:
            raise ValueError("Distinct FlexTensor managers cannot share a compiled-offload manager ID")
        manager_ids.add(manager_id)
        units = _manager_units(manager)
        if _replan_mode(manager) and not _status(manager).can_request:
            raise RuntimeError("FlexTensor strategy replanning is unavailable; use a fresh worker")
        for module in units:
            if id(module) in discovered:
                raise ValueError("Distinct FlexTensor managers cannot share an offload unit")
            if id(module.compute) in computes:
                raise ValueError("FlexTensor offload shells cannot share a compute module")
            discovered[id(module)] = module
            computes.add(id(module.compute))
        boundaries.append(
            _ManagerBoundary(
                manager,
                manager_id,
                manager.model,
                _forward_signature(manager.model),
                tuple((module, module.compute, _forward_signature(module)) for module in units),
            )
        )
    targets = list(discovered.values()) if modules is None else _collect_modules(modules)
    if any(id(module) not in discovered for module in targets):
        raise ValueError("Every selected module must be returned by a supplied manager's offload_units()")
    registration = register_for_jit_tuning(
        [discovered[id(module)].compute for module in targets], device_management="external"
    )
    return _FlexTensorRegistration(registration, tuple(boundaries))


def _forward_signature(module: torch.nn.Module) -> tuple[object, object]:
    """Identify a prepared forward without relying on transient bound-method objects."""
    forward = module.forward
    if type(forward) is MethodType:
        return forward.__self__, forward.__func__
    return None, forward


def _check_residency_boundary(boundary: _ManagerBoundary, units: tuple[FlexTensorUnit, ...]) -> None:
    """Require the same live public model, shells, computes and prepared shell forwards."""
    manager = boundary.manager
    if _manager_id(manager) != boundary.manager_id or manager.model is not boundary.model:
        raise RuntimeError("FlexTensor strategy replanning replaced the manager or model identity; use a fresh worker")
    if any(
        actual is not expected
        for actual, expected in zip(_forward_signature(manager.model), boundary.forward, strict=True)
    ):
        raise RuntimeError("FlexTensor strategy replanning replaced the model's prepared forward; use a fresh worker")
    if len(units) != len(boundary.units) or any(
        shell is not original
        or shell.compute is not compute
        or any(actual is not expected for actual, expected in zip(_forward_signature(shell), forward, strict=True))
        for shell, (original, compute, forward) in zip(units, boundary.units, strict=True)
    ):
        raise RuntimeError(
            "FlexTensor strategy replanning replaced a residency shell or prepared forward; use a fresh worker"
        )


def _registration_boundary(manager: Any, registration: JITRegistration) -> _ManagerBoundary:
    """Find the original public owner without silently accepting replaced compute."""
    if not isinstance(registration, _FlexTensorRegistration):
        raise TypeError("registration must come from register_flextensor_units()")
    boundary = next((item for item in registration._boundaries if item.manager is manager), None)
    if boundary is None:
        raise ValueError("manager must be one of the managers supplied to register_flextensor_units()")
    return boundary


def replan_flextensor(manager: Any, registration: JITRegistration, run_forward: Callable[[], Any]) -> None:
    """Measure and apply one manager's strategy after the selected AITune targets settle.

    Each ``run_forward()`` invocation must execute exactly one complete forward of this
    manager's model with representative inputs. The callback is invoked exactly as many
    times as FlexTensor requests; there is no hidden tuning or readiness loop. The caller
    must warm every route first, including lazy compilation, so measurements use stable
    tuned forwards. A selected eager backend is a real tuned selection; dry-run simulations,
    skipped targets and native eager fallbacks do not request replanning on their own.

    Pass the registration returned by :func:`register_flextensor_units`. Successful
    completion requires FlexTensor to report ``state='done', applied=True`` and preserve
    its original public ownership, prepared forwards and AITune's installed dispatch.
    Provider and callback exceptions propagate.
    A failed or incomplete replan requires a fresh worker; this helper provides no rollback,
    recovery or reset API.
    """
    _manager_id(manager)
    units = _manager_units(manager)
    if not _replan_mode(manager):
        raise RuntimeError(
            "FlexTensor retained-profile mode cannot replan because compiled_replan_active is false; "
            "keep the existing strategy and do not call replan_flextensor()"
        )
    boundary = _registration_boundary(manager, registration)
    _check_residency_boundary(boundary, units)
    unit_ids = {id(module.compute) for module in units}
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
    forwards = [(patched, _tuned_forward_signature(patched)) for patched in tuned]
    count = manager.request_strategy_replan()
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise RuntimeError("FlexTensor request_strategy_replan() must return a positive integer forward count")
    for _ in range(count):
        run_forward()
    status = _status(manager)
    if status.state != "done" or status.applied is not True:
        raise RuntimeError("FlexTensor did not complete and apply strategy replanning; use a fresh worker")
    _check_residency_boundary(boundary, _manager_units(manager))
    if any(
        any(
            actual is not expected
            for actual, expected in zip(_tuned_forward_signature(patched), signature, strict=True)
        )
        for patched, signature in forwards
    ):
        raise RuntimeError("FlexTensor strategy replanning replaced an installed AITune forward; use a fresh worker")
