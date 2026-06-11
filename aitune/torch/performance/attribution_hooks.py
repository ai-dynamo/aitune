# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hook-based instrumentation for performance profile regions (AOT and untuned)."""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from aitune.torch.jit.patcher import Patcher
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.performance.context import (
    AOT_MODULE_REGION_PREFIX,
    UNTUNED_MODULE_REGION_PREFIX,
)
from aitune.torch.performance.utils import _qualified_type_name

# State is stored as a per-module attribute, which is process-wide (not thread-local).
# Safe because PyTorch inference is sequential at the Python level (GIL) and concurrent
# inference services use separate model instances per worker. Multi-threaded forward on
# the same module would race; switch to threading.local keyed by id(module) if needed.
#
# Forward hooks AND method wrappers share this single attribute. A method that
# internally calls ``__call__`` (or vice versa) on the same module finds the
# attribute already set and skips opening a nested context — only the outermost
# entry point emits a record_function span, so nested same-module work is never
# double-counted. The first entry's annotation name wins.
_CONTEXT_ATTR = "_aitune_performance_record_function"


@dataclass(frozen=True)
class _RegionTarget:
    """One module to instrument with a record_function annotation during capture.

    ``annotation_prefix`` distinguishes the kind of region: AOT-managed modules
    use :data:`AOT_MODULE_REGION_PREFIX`, top-level untuned modules use
    :data:`UNTUNED_MODULE_REGION_PREFIX`. The full annotation name is
    ``f"{annotation_prefix}{path}"``.
    """

    path: str
    module: nn.Module
    module_type: str
    annotation_prefix: str


class _ActiveRegion:
    """Per-module recording state. Holds the open record_function context and a depth counter."""

    __slots__ = ("ctx", "depth")

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.depth = 1


def _discover_aot_targets() -> list[_RegionTarget]:
    """Discover AITune-managed AOT regions from the global module registry.

    Each registered ``Module`` wrapper becomes one target whose path is the
    wrapper's user-given name. The hook is registered on the wrapper proxy
    itself; ``wrapt`` forwards the registration to ``wrapper.__wrapped__`` so
    the hook fires from the standard ``nn.Module.__call__`` machinery when the
    wrapper is invoked.

    All registered wrappers are instrumented unconditionally. If two wrappers
    are statically nested (one's underlying module is a descendant of another's
    subtree), they may emit overlapping spans on different modules during
    measurement. Whether that overlap actually happens depends on what the
    profiled workload calls — only the wrappers that fire in a given run can
    overlap. Event-time suppression in :meth:`_RuntimeProfile._classify_region_events`
    handles the dynamic case correctly: a nested AOT event observed as a
    descendant of another AOT event in the same profiled run is skipped from
    aggregation; events fired independently (e.g. the inner wrapper invoked
    directly while the outer never ran) are counted normally.
    """
    targets: list[_RegionTarget] = []
    for wrapper in MODULE_REGISTRY.modules.values():
        wrapped = getattr(wrapper, "__wrapped__", wrapper)
        targets.append(
            _RegionTarget(
                path=wrapper.name,
                module=wrapper,
                module_type=_qualified_type_name(wrapped),
                annotation_prefix=AOT_MODULE_REGION_PREFIX,
            )
        )
    return targets


def _discover_untuned_targets(obj: Any) -> list[_RegionTarget]:
    """Discover untuned module roots from obj.

    For ``nn.Module`` objects, traversal starts from ``obj.named_children()``.
    For other objects (pipeline-like), it starts from direct ``vars(obj)`` attributes
    that are ``nn.Module`` instances. Plain Python ``list`` / ``dict`` attributes are
    not entered; ``nn.ModuleList`` / ``nn.ModuleDict`` are entered transparently
    because they're never invoked as modules themselves (their ``forward`` raises),
    so their children are what get attributed (see :func:`_is_transparent_container`).

    From each starting point, discovery walks the subtree to find the topmost
    untuned modules:

    - If a module is itself AITune-managed (AOT-wrapped or JIT-patched), it is skipped
      (its time is already attributed via the wrapper).
    - If a module is a transparent container, discovery recurses into its children.
    - If a module is clean but contains an AOT/JIT-managed descendant, discovery
      recurses into its children so siblings of the managed module are instrumented
      individually at their natural depth.
    - Otherwise the module is instrumented as one untuned region.

    Results are deduped by ``id(module)``: if two attribute paths reach the same
    underlying instance (e.g. ``self.vae`` and an alias ``self.legacy_vae = self.vae``),
    the first-seen path wins. Aliases would otherwise double-instrument the same
    methods and corrupt restoration on teardown.
    """
    aot_managed_ids = _aot_managed_module_ids()
    candidates = _enumerate_candidates(obj)

    targets: list[_RegionTarget] = []
    for path, module in candidates:
        targets.extend(_discover_in_subtree(module, path, aot_managed_ids))

    seen_ids: set[int] = set()
    unique: list[_RegionTarget] = []
    for region_target in targets:
        if id(region_target.module) in seen_ids:
            continue
        seen_ids.add(id(region_target.module))
        unique.append(region_target)
    return unique


def _discover_in_subtree(module: nn.Module, path: str, aot_managed_ids: set[int]) -> list[_RegionTarget]:
    """Walk one subtree and return targets for the topmost clean (non-managed) submodules.

    Transparent containers (``nn.ModuleList``, ``nn.ModuleDict``) are never invoked
    as modules — caller code iterates their entries — so installing a hook on the
    container would never fire and the contained compute would land in residual.
    Recurse through them instead of attributing them as a single region.
    """
    if _is_managed(module, aot_managed_ids):
        return []
    if not _is_transparent_container(module) and not _has_managed_descendant(module, aot_managed_ids):
        return [
            _RegionTarget(
                path=path,
                module=module,
                module_type=_qualified_type_name(module),
                annotation_prefix=UNTUNED_MODULE_REGION_PREFIX,
            )
        ]

    targets: list[_RegionTarget] = []
    for name, child in module.named_children():
        child_path = f"{path}.{name}"
        targets.extend(_discover_in_subtree(child, child_path, aot_managed_ids))
    return targets


def _is_transparent_container(module: nn.Module) -> bool:
    """Return True for ``nn.Module`` subclasses that are never themselves invoked as modules.

    ``ModuleList`` and ``ModuleDict`` raise ``NotImplementedError`` from their ``forward``
    by design — caller code iterates over their entries (``for layer in self.layers``)
    rather than calling them. Treating them as leaf attribution targets would install
    hooks that never fire.
    """
    return isinstance(module, (nn.ModuleList, nn.ModuleDict))


def _enumerate_candidates(obj: Any) -> list[tuple[str, nn.Module]]:
    """Return (path, module) tuples for the top-level entry points into obj.

    These are the starting points for subtree discovery — direct children for an
    ``nn.Module`` object, or direct attributes for a pipeline-like object.

    Note: when ``obj`` is itself an ``nn.Module``, its own methods are not
    instrumented — discovery starts one level down. Per-child breakdown is
    preserved at the cost of root-level method capture. To attribute root-level
    method entry points (e.g. ``model.generate``), wrap the model in a small
    pipeline-like container that holds it as an attribute and pass that as the
    object to profile. Removing this limitation requires hierarchical attribution
    (overlapping spans) which is tracked as Layer 2 follow-up.
    """
    if isinstance(obj, nn.Module):
        return list(obj.named_children())

    if hasattr(obj, "__dict__"):
        return [(name, value) for name, value in vars(obj).items() if isinstance(value, nn.Module)]

    return []


def _aot_managed_module_ids() -> set[int]:
    """Build the set of id(m) covering AITune AOT wrappers AND their wrapped subtrees.

    The wrapper proxy and the underlying ``nn.Module`` instance are distinct Python
    objects with different ids. Both must be included so that traversal correctly
    identifies an AOT-managed module whether it surfaces as the wrapper (via
    ``parent.named_children()``) or as the wrapped instance (via ``module.modules()``).
    """
    ids: set[int] = set()
    for wrapper in MODULE_REGISTRY.modules.values():
        ids.add(id(wrapper))
        wrapped = getattr(wrapper, "__wrapped__", None)
        if not isinstance(wrapped, nn.Module):
            continue
        for module in wrapped.modules():
            ids.add(id(module))
    return ids


def _is_managed(module: nn.Module, aot_managed_ids: set[int]) -> bool:
    """Return True if module itself is AOT-wrapped or JIT-patched."""
    if id(module) in aot_managed_ids:
        return True
    return any(patched.__wrapped__ is module for patched in Patcher.patched_modules_under(module))


def _has_managed_descendant(module: nn.Module, aot_managed_ids: set[int]) -> bool:
    """Return True if any strict descendant (excluding module itself) is AOT- or JIT-managed."""
    for descendant in module.modules():
        if descendant is module:
            continue
        if id(descendant) in aot_managed_ids:
            return True
    return any(patched.__wrapped__ is not module for patched in Patcher.patched_modules_under(module))


def _user_defined_method_names(module: nn.Module) -> list[str]:
    """Return method names visible on ``type(module)`` (including ancestors) beyond what ``nn.Module`` provides.

    Walks the full MRO via :func:`dir`, which is intentional — many framework
    entry points like HF Transformers' ``GenerationMixin.generate`` are defined
    on mixins rather than on the concrete model subclass. Filters out:

    - names also present in ``dir(nn.Module)`` (already covered by forward hooks or framework concerns)
    - dunder methods and leading-underscore names
    - the ``forward`` method (already hook-covered)
    - non-function descriptors (properties, classmethods, staticmethods, slot wrappers)

    Uses :func:`inspect.getattr_static` so accessing a property descriptor does not
    fire its underlying ``fget``.
    """
    cls = type(module)
    inherited = set(dir(nn.Module))
    names: list[str] = []
    for name in dir(cls):
        if name.startswith("_") or name in inherited or name == "forward":
            continue
        try:
            attr = inspect.getattr_static(cls, name)
        except AttributeError:
            continue
        if inspect.isfunction(attr):
            names.append(name)
    return names


class _RegionInstaller:
    """Context manager installing forward hooks and method wrappers on a list of region targets.

    For every target the installer registers pre/post forward hooks. For untuned
    targets it additionally monkey-patches the underlying module's user-defined
    methods (e.g. ``vae.decode``, ``model.generate``) with
    ``torch.profiler.record_function`` wrappers. Untuned targets are leaves by
    construction (discovery does not emit a parent that contains another emitted
    region), so a method span cannot overlap a sibling region. AOT targets get
    forward hooks only — wrapping their user methods would overlap their own
    forward-hook span when the method internally calls ``forward``.

    Forward hooks and method wrappers share ``_CONTEXT_ATTR`` on each module. The
    first entry point (whether forward or a wrapped method) opens a
    ``record_function`` span and stashes it on the module. Subsequent entries on
    the same module — for example ``decode`` calling ``self(...)``, or ``forward``
    calling ``self.decode(...)`` — find the attribute already set, bump a depth
    counter and return without opening a nested context. This prevents
    same-module double-counting in the per-run aggregation. The outer entry's
    annotation name is what appears in the trace.
    Post-hook (``always_call=True``) and the method wrapper's ``finally`` both
    decrement the depth and close+clear the context when it reaches zero.

    Pre-existing instance-level method overrides are preserved: the installer
    records whether a name was already in ``vars(module)`` before patching and
    restores it on ``__exit__`` (vs. ``delattr`` for class-defined methods).

    Caveat: monkey-patching happens at the instance attribute level, so a method
    reference bound *before* the installer enters (e.g. ``saved = pipe.vae.decode``)
    will bypass the wrapper if called later. Pipeline code that resolves the
    attribute at call site (``self.vae.decode(x)``) is captured normally.

    On ``__exit__`` the installer restores patched methods, removes hook handles,
    and sweeps any leftover context attributes. Cleanup failures are collected
    and re-raised as a single ``RuntimeError``.
    """

    def __init__(self, targets: list[_RegionTarget]) -> None:
        self._targets = targets
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        # (underlying_module, method_name, original_callable, was_instance_attr_before_patch)
        self._method_wraps: list[tuple[nn.Module, str, Any, bool]] = []

    def __enter__(self) -> _RegionInstaller:
        for target in self._targets:
            pre_handle = target.module.register_forward_pre_hook(self._make_pre_hook(target))
            post_handle = target.module.register_forward_hook(self._make_post_hook(), always_call=True)
            self._handles.append(pre_handle)
            self._handles.append(post_handle)

            if target.annotation_prefix != UNTUNED_MODULE_REGION_PREFIX:
                continue
            underlying = _underlying_module(target.module)
            for method_name in _user_defined_method_names(underlying):
                was_instance_attr = method_name in vars(underlying)
                original = getattr(underlying, method_name)
                if not callable(original):
                    # Instance attribute shadows the class method with a non-callable
                    # (e.g. someone set `module.decode = "disabled"`). Wrapping would
                    # silently change runtime behavior and explode on call. Leave alone.
                    continue
                annotation = f"{target.annotation_prefix}{target.path}.{method_name}"
                setattr(underlying, method_name, self._make_method_wrapper(underlying, annotation, original))
                self._method_wraps.append((underlying, method_name, original, was_instance_attr))
        return self

    def __exit__(self, *exc: Any) -> None:
        cleanup_errors: list[Exception] = []
        self._restore_methods(cleanup_errors)
        self._method_wraps.clear()
        self._remove_handles(cleanup_errors)
        self._sweep_active_contexts(cleanup_errors)
        if cleanup_errors:
            # Surface the cleanup failure — it indicates a real bug (e.g. a hook
            # handle that refused to remove). If the with-block was already
            # propagating an exception, Python's implicit ``__context__`` chains
            # it alongside this RuntimeError in the traceback. Don't use
            # ``raise ... from`` because the two failures are independent — the
            # workload exception didn't cause the cleanup failure, they just
            # both happened.
            raise RuntimeError(f"Failed to clean up region hooks: {cleanup_errors!r}")

    def _restore_methods(self, cleanup_errors: list[Exception]) -> None:
        for underlying, method_name, original, was_instance_attr in self._method_wraps:
            try:
                if was_instance_attr:
                    setattr(underlying, method_name, original)
                else:
                    delattr(underlying, method_name)
            except AttributeError:
                pass
            except Exception as error:
                cleanup_errors.append(error)

    def _remove_handles(self, cleanup_errors: list[Exception]) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception as error:
                cleanup_errors.append(error)
        self._handles.clear()

    def _sweep_active_contexts(self, cleanup_errors: list[Exception]) -> None:
        """Close any leftover region context attributes on every target's underlying module."""
        for target in self._targets:
            _close_and_delete_context(target.module, _CONTEXT_ATTR, cleanup_errors)
            underlying = _underlying_module(target.module)
            if underlying is not target.module:
                _close_and_delete_context(underlying, _CONTEXT_ATTR, cleanup_errors)

    def _make_pre_hook(self, target: _RegionTarget) -> Any:
        annotation = f"{target.annotation_prefix}{target.path}"

        def pre_hook(module: nn.Module, args: tuple[Any, ...]) -> None:
            active = getattr(module, _CONTEXT_ATTR, None)
            if active is not None:
                active.depth += 1
                return
            ctx = torch.profiler.record_function(annotation)
            ctx.__enter__()
            setattr(module, _CONTEXT_ATTR, _ActiveRegion(ctx))

        return pre_hook

    def _make_post_hook(self) -> Any:
        def post_hook(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
            active = getattr(module, _CONTEXT_ATTR, None)
            if active is None:
                return
            active.depth -= 1
            if active.depth > 0:
                return
            try:
                active.ctx.__exit__(None, None, None)
            finally:
                try:
                    delattr(module, _CONTEXT_ATTR)
                except AttributeError:
                    pass

        return post_hook

    def _make_method_wrapper(self, underlying: nn.Module, annotation: str, original: Any) -> Any:
        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            active = getattr(underlying, _CONTEXT_ATTR, None)
            if active is not None:
                active.depth += 1
                try:
                    return original(*args, **kwargs)
                finally:
                    active.depth -= 1
            ctx = torch.profiler.record_function(annotation)
            ctx.__enter__()
            active = _ActiveRegion(ctx)
            setattr(underlying, _CONTEXT_ATTR, active)
            try:
                return original(*args, **kwargs)
            finally:
                active.depth -= 1
                if active.depth <= 0:
                    try:
                        active.ctx.__exit__(None, None, None)
                    finally:
                        try:
                            delattr(underlying, _CONTEXT_ATTR)
                        except AttributeError:
                            pass

        return wrapper


def _underlying_module(module: nn.Module) -> nn.Module:
    """Return the underlying ``nn.Module``, unwrapping an AITune ``wrapt`` proxy if present."""
    wrapped = getattr(module, "__wrapped__", None)
    if isinstance(wrapped, nn.Module):
        return wrapped
    return module


def _close_and_delete_context(holder: Any, attr_name: str, cleanup_errors: list[Exception]) -> None:
    """Close any active region context stored on ``holder`` at ``attr_name`` and delete the attribute."""
    active = getattr(holder, attr_name, None)
    if active is None:
        return
    try:
        active.ctx.__exit__(None, None, None)
    except Exception as error:
        cleanup_errors.append(error)
    try:
        delattr(holder, attr_name)
    except AttributeError:
        pass
    except Exception as error:
        cleanup_errors.append(error)
