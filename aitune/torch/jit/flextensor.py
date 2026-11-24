# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register native FlexTensor offload units and synchronously replan their manager."""

from collections.abc import Callable, Iterable
from types import MethodType
from typing import Any, NamedTuple

import torch
import wrapt

from aitune.torch.jit.patched_module import PatchedModule
from aitune.torch.jit.registration import JITRegistration, _collect_modules, register_for_jit_tuning
from aitune.torch.module.tuned_module import TunedModule


class _ManagerBoundary(NamedTuple):
    """Original public ownership and prepared forwards for one manager."""

    manager: Any
    manager_id: int
    model: torch.nn.Module
    forward: tuple[object, object]
    units: tuple[tuple[torch.nn.Module, tuple[object, object]], ...]


class _FlexTensorRegistration(JITRegistration):
    """Retain manager ownership alongside the ordinary live registration view."""

    __slots__ = ("_boundaries",)

    def __init__(self, registration: JITRegistration, boundaries: tuple[_ManagerBoundary, ...]) -> None:
        """Retain the unchanged core registration's native targets and owners."""
        super().__init__(registration._patched_modules)
        self._boundaries = boundaries


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


def _replan_mode(manager: Any) -> bool:
    """Validate the active public compiled-offload mode without requesting a replan."""
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


def _manager_id(manager: Any) -> int:
    """Read the public identity used to reject ambiguous ownership."""
    manager_id = getattr(manager, "compiled_offload_manager_id", None)
    if not isinstance(manager_id, int) or isinstance(manager_id, bool):
        raise RuntimeError("FlexTensor must expose an integer compiled_offload_manager_id")
    return manager_id


def _manager_units(manager: Any) -> tuple[torch.nn.Module, ...]:
    """Resolve live original modules after public discovery and profiling finish."""
    if getattr(getattr(manager, "phase", None), "value", None) != "inference":
        raise RuntimeError("FlexTensor must finish discovery and profiling and reach inference before registration")
    _replan_mode(manager)
    if not callable(getattr(manager, "offload_units", None)):
        raise RuntimeError("FlexTensor must expose public offload_units()")
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


def _forward_signature(forward: Callable[..., Any]) -> tuple[object, object]:
    """Identify a forward without relying on transient bound-method objects."""
    if type(forward) is MethodType:
        return forward.__self__, forward.__func__
    return None, forward


def _same_forward(forward: Callable[..., Any], expected: tuple[object, object]) -> bool:
    """Compare the owner and implementation of a prepared forward by identity."""
    return all(actual is original for actual, original in zip(_forward_signature(forward), expected, strict=True))


def _dispatch_signature(patched: PatchedModule) -> tuple[object, object]:
    """Require AITune's own proxy around the captured FlexTensor forward."""
    forward = patched.__wrapped__.forward
    router = getattr(forward, "_self_wrapper", None)
    router_function = getattr(router, "__func__", None)
    if (
        type(forward) is not wrapt.FunctionWrapper
        or getattr(router, "__self__", None) is not patched
        or router_function is not patched._forward_router.__func__
        or forward.__wrapped__ is not patched._original_forward
    ):
        raise RuntimeError("FlexTensor strategy replanning replaced an installed AITune forward; use a fresh worker")
    return forward.__wrapped__, router_function


def _tuned_forward_signature(patched: PatchedModule) -> tuple[object, object, object]:
    """Identify tuned dispatch across proxies recreated by ordinary inference."""
    original, router = _dispatch_signature(patched)
    if not isinstance(patched._wrapper, TunedModule):
        raise RuntimeError("FlexTensor strategy replanning replaced an installed AITune forward; use a fresh worker")
    return original, router, patched._wrapper


def register_flextensor_units(
    managers: Iterable[Any], *, modules: Iterable[torch.nn.Module] | None = None
) -> JITRegistration:
    """Register original public offload units, or an explicit subset, once per worker.

    Prepare the model with public FlexTensor APIs and finish discovery/profiling before
    registration. The final live ``manager.offload_units()`` modules are registered directly
    using ``device_management="external"``. No module, parameter or checkpoint path is added.
    ``modules`` optionally selects those same original objects in the requested order.

    Identity-deduplicated managers and units retain discovery order. AITune's existing
    external-placement and backend validation apply unchanged. Native repeated-unit tuning
    uses FlexTensor's per-unit compilation and replay support.

    External compilation/replan mode must expose an available public replan request. The
    retained view-profile mode also works; it keeps FT's pre-AITune strategy and must not
    call :func:`replan_flextensor`. The caller records representative model forwards, tunes
    and warms every serving route before replanning. FlexTensor is optional and not imported.
    """
    unique_managers = tuple({id(manager): manager for manager in managers}.values())
    if not unique_managers:
        raise ValueError("managers must contain at least one FlexTensor manager")
    discovered: dict[int, torch.nn.Module] = {}
    manager_ids: set[int] = set()
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
            discovered[id(module)] = module
        boundaries.append(
            _ManagerBoundary(
                manager,
                manager_id,
                manager.model,
                _forward_signature(manager.model.forward),
                tuple((module, _forward_signature(module.forward)) for module in units),
            )
        )
    targets = list(discovered.values()) if modules is None else _collect_modules(modules)
    if any(id(module) not in discovered for module in targets):
        raise ValueError("Every selected module must be returned by a supplied manager's offload_units()")
    registration = register_for_jit_tuning(targets, device_management="external")
    return _FlexTensorRegistration(registration, tuple(boundaries))


def _check_unit_forward(
    module: torch.nn.Module, original: tuple[object, object], patched: PatchedModule | None
) -> None:
    """Require the original prepared forward beneath any active AITune dispatcher."""
    if patched is None or patched._state.value in ("eager", "skipped"):
        if not _same_forward(module.forward, original):
            raise RuntimeError("FlexTensor strategy replanning replaced a prepared forward; use a fresh worker")
    if patched is not None:
        if not _same_forward(patched._original_forward, original):
            raise RuntimeError("FlexTensor strategy replanning replaced a prepared forward; use a fresh worker")
        if patched._state.value not in ("eager", "skipped"):
            _dispatch_signature(patched)


def _check_residency_boundary(
    boundary: _ManagerBoundary, units: tuple[torch.nn.Module, ...], registration: JITRegistration
) -> None:
    """Require the same provider tree and prepared forwards behind native JIT dispatch."""
    manager = boundary.manager
    if _manager_id(manager) != boundary.manager_id or manager.model is not boundary.model:
        raise RuntimeError("FlexTensor strategy replanning replaced the manager or model identity; use a fresh worker")
    if len(units) != len(boundary.units) or any(
        actual is not original for actual, (original, _) in zip(units, boundary.units, strict=True)
    ):
        raise RuntimeError("FlexTensor strategy replanning replaced an offload unit; use a fresh worker")
    patched_by_id = {id(patched.__wrapped__): patched for patched in registration._patched_modules}
    for module, original in boundary.units:
        _check_unit_forward(module, original, patched_by_id.get(id(module)))
    if id(manager.model) not in {id(module) for module in units} and not _same_forward(
        manager.model.forward, boundary.forward
    ):
        raise RuntimeError("FlexTensor strategy replanning replaced the model's prepared forward; use a fresh worker")


def _registration_boundary(manager: Any, registration: JITRegistration) -> _ManagerBoundary:
    """Find the original public owner without silently accepting replaced units."""
    if not isinstance(registration, _FlexTensorRegistration):
        raise TypeError("registration must come from register_flextensor_units()")
    boundary = next((item for item in registration._boundaries if item.manager is manager), None)
    if boundary is None:
        raise ValueError("manager must be one of the managers supplied to register_flextensor_units()")
    return boundary


def replan_flextensor(manager: Any, registration: JITRegistration, run_forward: Callable[[], Any]) -> None:
    """Measure and apply an offload strategy after selected AITune targets settle.

    Each callback invocation must execute exactly one complete manager-model forward with
    representative inputs. The caller must first warm every route, including lazy compilation.
    A selected eager backend is a real tuned result. Dry-run simulations, skipped targets and
    native eager fallbacks do not request replanning on their own.

    Completion requires public ``state='done', applied=True``, unchanged original ownership
    and prepared forwards, and the same installed tuned dispatch. Provider and callback errors
    propagate. Failure requires a fresh worker; no rollback, recovery or reset is provided.
    """
    _manager_id(manager)
    units = _manager_units(manager)
    if not _replan_mode(manager):
        raise RuntimeError(
            "FlexTensor retained-profile mode cannot replan because compiled_replan_active is false; "
            "keep the existing strategy and do not call replan_flextensor()"
        )
    boundary = _registration_boundary(manager, registration)
    _check_residency_boundary(boundary, units, registration)
    unit_ids = {id(module) for module in units}
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
    _check_residency_boundary(boundary, _manager_units(manager), registration)
    if any(
        current is not original
        for patched, expected in forwards
        for current, original in zip(_tuned_forward_signature(patched), expected, strict=True)
    ):
        raise RuntimeError("FlexTensor strategy replanning replaced an installed AITune forward; use a fresh worker")
