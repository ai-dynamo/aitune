# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module for inspecting PyTorch models and tracking their execution."""

import atexit
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from functools import wraps
from typing import TypeVar

import torch

from aitune.torch.config import AITuneMode
from aitune.torch.jit.config import JITMode, config
from aitune.torch.jit.inspect_module import InspectModule
from aitune.torch.jit.patched_module import ModuleState as PatchedModuleState
from aitune.torch.jit.patched_module import PatchedModule
from aitune.torch.tune_data.report_models import TuneRunReport
from aitune.torch.tune_data.reporting import (
    has_active_report,
    report_inspection_details,
    report_tune_run_end,
    report_tune_run_start,
)

T = TypeVar("T")


# Built-in exclusions the JIT patcher always applies. User-supplied entries on
# ``jit_config.patch_exclude`` add to these — they cannot be removed via config to
# avoid breaking tuning by accident (e.g. dropping ``torch_tensorrt`` leads to
# wrapt-decorated forwards in the compiled gm and crashes ``torch_tensorrt.save``).
_DEFAULT_PATCH_EXCLUDE: tuple[str, ...] = (
    "torch.jit",
    "torch._inductor",
    "torch._dynamo",
    "torch.fx",
    "torch.export",
    "torch_tensorrt",
    "torch.nn.modules.container.ModuleList",
    "torch.nn.modules.container.ModuleDict",
    "torch.nn.modules.container.Sequential",
    "torch.nn.modules.container.ParameterList",
    "torch.nn.modules.container.ParameterDict",
)


class Patcher:
    """Patcher class.

    Allows intercepting creating torch modules to register them with PatchedModule.
    """

    _patched_modules: list[PatchedModule | InspectModule] = []
    _intercepted_classes: set[type[torch.nn.Module]] = set()  # for tracking purposes
    _original_module_init: Callable | None = None
    _torch_patched: bool = False
    _exit_handler: Callable | None = None
    _session_mode: JITMode | None = None
    _session_report: TuneRunReport | None = None
    _cleanup_pending: bool = False

    @classmethod
    def _ensure_session(cls) -> None:
        """Initialize reporting and process-exit handling for one JIT session."""
        cls._raise_if_cleanup_pending()
        cls._raise_if_session_mode_changed()

        cls._session_mode = config.mode
        if cls._exit_handler is None:
            cls._exit_handler = (
                InspectModule.on_python_exit if config.mode == JITMode.INSPECT else PatchedModule.on_python_exit
            )
            atexit.register(cls._exit_handler)

        if config.mode != JITMode.INSPECT and not has_active_report():
            cls._session_report = report_tune_run_start(AITuneMode.JIT)

    @classmethod
    def _end_session_report(cls, exception: BaseException | None = None) -> None:
        """Finalize only the report created by this JIT session, if still active."""
        if cls._session_report is None:
            return
        try:
            report_tune_run_end(exception=exception, expected_report=cls._session_report)
        finally:
            cls._session_report = None

    @classmethod
    def _raise_if_cleanup_pending(cls) -> None:
        """Prevent a new tuning session from reusing partially cleaned wrappers."""
        if cls._cleanup_pending:
            raise RuntimeError("JIT cleanup is incomplete; retry jit_reset() before registering or patching modules")

    @classmethod
    def _register_module(
        cls,
        module: torch.nn.Module,
        *,
        explicit_head: bool = False,
    ) -> PatchedModule | InspectModule:
        """Create and retain one module wrapper."""
        if config.mode == JITMode.INSPECT:
            patched_module: PatchedModule | InspectModule = InspectModule(module)
        else:
            patched_module = PatchedModule(module, explicit_head=explicit_head)
        cls._patched_modules.append(patched_module)
        return patched_module

    @classmethod
    def patch_torch(cls):
        """Patch torch.nn.Module to track execution."""
        cls._raise_if_cleanup_pending()
        if any(isinstance(module, PatchedModule) and module.explicit_head for module in cls._patched_modules):
            raise RuntimeError(
                "Global JIT constructor interception cannot run while explicit JIT registrations are active"
            )
        cls._ensure_session()
        if cls._torch_patched:
            return

        if cls._original_module_init is None:
            cls._original_module_init = torch.nn.Module.__init__

        def _patched_init(module, *args, **kwargs):
            cls._raise_if_session_mode_changed()
            cls._original_module_init(module, *args, **kwargs)
            if cls._is_allowed_to_tune(module):
                cls._intercepted_classes.add(module.__class__)
                cls._register_module(module)

        torch.nn.Module.__init__ = _patched_init
        cls._torch_patched = True

    @classmethod
    def register_modules(cls, modules: Sequence[torch.nn.Module]) -> tuple[PatchedModule, ...]:
        """Register validated, pre-existing modules as independent JIT tuning heads."""
        cls.validate_explicit_registration_session()

        trees = cls._validate_registration_targets(modules)
        resolved = [cls._find_reusable_registration(module, tree) for module, tree in zip(modules, trees, strict=True)]

        previous_module_counter = PatchedModule.module_counter
        previous_session_mode = cls._session_mode
        previous_exit_handler = cls._exit_handler
        previous_session_report = cls._session_report
        newly_registered: list[PatchedModule] = []
        try:
            cls._ensure_session()
            for index, module in enumerate(modules):
                if resolved[index] is not None:
                    continue
                wrapper = cls._register_module(module, explicit_head=True)
                assert isinstance(wrapper, PatchedModule)
                newly_registered.append(wrapper)
                resolved[index] = wrapper

            assert all(wrapper is not None for wrapper in resolved)
            return tuple(wrapper for wrapper in resolved if wrapper is not None)
        except Exception as error:
            try:
                cls._rollback_explicit_registrations(newly_registered)
            except Exception as cleanup_error:
                error.add_note(f"JIT registration cleanup failed; retry jit_reset(): {cleanup_error}")
            finally:
                if not cls._cleanup_pending:
                    PatchedModule.module_counter = previous_module_counter
                try:
                    cls._rollback_explicit_registration_session(
                        previous_session_mode,
                        previous_exit_handler,
                        previous_session_report,
                        error,
                    )
                except Exception as report_error:
                    error.add_note(f"JIT registration report finalization failed: {report_error}")
            raise

    @classmethod
    def validate_explicit_registration_session(cls) -> None:
        """Reject explicit registration while an incompatible JIT path is active."""
        cls._raise_if_cleanup_pending()
        cls._raise_if_session_mode_changed()
        if config.mode == JITMode.INSPECT:
            raise RuntimeError("register_for_jit_tuning() does not support JITMode.INSPECT")
        if cls._torch_patched:
            raise RuntimeError(
                "register_for_jit_tuning() cannot run while global JIT constructor interception is active"
            )
        if any(isinstance(module, PatchedModule) and not module.explicit_head for module in cls._patched_modules):
            raise RuntimeError(
                "register_for_jit_tuning() cannot run after automatic JIT discovery; call jit_reset() first"
            )

    @classmethod
    def _raise_if_session_mode_changed(cls) -> None:
        """Require an explicit reset before changing the active JIT mode."""
        if cls._session_mode is not None and cls._session_mode != config.mode:
            raise RuntimeError("Call jit_reset() before switching JIT modes")

    @classmethod
    def _rollback_explicit_registrations(cls, wrappers: Sequence[PatchedModule]) -> None:
        """Undo wrappers created by an explicit-registration transaction."""
        first_error: Exception | None = None
        for wrapper in reversed(wrappers):
            try:
                wrapper._update_state(PatchedModuleState.DETACHED)
                wrapper._unpatch()
            except Exception as error:
                first_error = first_error or error
        if first_error is not None:
            cls._cleanup_pending = True
            raise first_error

    @classmethod
    def _rollback_explicit_registration_session(
        cls,
        previous_session_mode: JITMode | None,
        previous_exit_handler: Callable | None,
        previous_session_report: TuneRunReport | None,
        error: Exception,
    ) -> None:
        """Restore session globals when an explicit-registration transaction fails."""
        try:
            if cls._session_report is not previous_session_report:
                cls._end_session_report(exception=error)
        finally:
            if not cls._cleanup_pending:
                try:
                    if cls._exit_handler is not previous_exit_handler and cls._exit_handler is not None:
                        atexit.unregister(cls._exit_handler)
                finally:
                    cls._exit_handler = previous_exit_handler
                    cls._session_mode = previous_session_mode
                    cls._session_report = previous_session_report

    @classmethod
    def _validate_registration_targets(cls, modules: Sequence[torch.nn.Module]) -> list[set[int]]:
        """Validate exclusions and disjoint ownership without changing module state."""
        trees = [cls._module_tree_ids(module) for module in modules]
        for index, module in enumerate(modules):
            if not cls._is_allowed_to_tune(module):
                raise ValueError(f"Explicit JIT target is excluded from tuning: {cls._module_label(module)}")
            cls._raise_if_target_overlaps(module, trees[index], modules[:index], trees[:index])
        return trees

    @classmethod
    def _raise_if_target_overlaps(
        cls,
        module: torch.nn.Module,
        tree: set[int],
        previous_modules: Sequence[torch.nn.Module],
        previous_trees: Sequence[set[int]],
    ) -> None:
        """Reject overlapping ownership trees within one registration call."""
        for previous_module, previous_tree in zip(previous_modules, previous_trees, strict=True):
            if tree & previous_tree:
                raise ValueError(
                    "Explicit JIT targets must have disjoint module ownership trees: "
                    f"{cls._module_label(previous_module)} and {cls._module_label(module)}"
                )

    @classmethod
    def _find_reusable_registration(
        cls,
        module: torch.nn.Module,
        tree: set[int],
    ) -> PatchedModule | None:
        """Return an identical registration or reject overlap with existing wrappers."""
        reusable: PatchedModule | None = None
        for existing in cls._patched_modules:
            existing_module = existing.__wrapped__
            if not tree & cls._module_tree_ids(existing_module):
                continue
            if existing_module is module and isinstance(existing, PatchedModule) and existing.explicit_head:
                reusable = existing
                continue
            raise ValueError(
                "Explicit JIT target overlaps an existing JIT registration: "
                f"{cls._module_label(module)} and {cls._module_label(existing_module)}"
            )
        return reusable

    @staticmethod
    def _module_tree_ids(module: torch.nn.Module) -> set[int]:
        """Return identity keys for a module's ownership tree."""
        return {id(owned_module) for owned_module in module.modules()}

    @staticmethod
    def _module_label(module: torch.nn.Module) -> str:
        """Return a useful target label for validation errors."""
        module_class = module.__class__
        return f"{module_class.__module__}.{module_class.__qualname__} (id={id(module)})"

    @classmethod
    def enable_tune_deferred(cls, enable: bool = True):
        """Enable or disable tuning for recorded modules in deferred mode.

        Call this after running at least one full forward pass through the pipeline so that every
        module has collected the samples it needs. Tuning runs from the next normal forward path
        instead of this method, which preserves pipeline-managed device placement.
        """
        if enable:
            report_inspection_details([
                module.inspection_report() for module in cls._patched_modules if isinstance(module, PatchedModule)
            ])
        PatchedModule.deferred_tuning_enabled = enable

    @classmethod
    def patched_modules_under(cls, root: torch.nn.Module) -> list[PatchedModule | InspectModule]:
        """Return patched modules registered under a root module's ownership tree.

        AITune's PatchedModule children model the observed call hierarchy, which can miss
        registered submodules that are not executed for the recorded samples. Backend build
        paths still receive and may copy/export the full nn.Module ownership tree, so cleanup
        at backend boundaries must operate on root.modules() instead of only observed children.
        """
        module_ids = {id(module) for module in root.modules()}
        return [module for module in cls._patched_modules if id(module.__wrapped__) in module_ids]

    @classmethod
    def unpatch_module(cls, module: PatchedModule | InspectModule):
        """Unpatch a module.

        Args:
            module: The module to unpatch.
        """
        if module in cls._patched_modules:
            cls._patched_modules.remove(module)

    @classmethod
    def unpatch_torch(cls, unpatch_modules: bool = False):
        """Unpatch torch.nn.Module.

        Args:
            unpatch_modules: if True, unpatch all modules, otherwise only torch.nn.Module
        """
        try:
            if unpatch_modules:
                first_error: Exception | None = None
                for patched_module in list(cls._patched_modules):
                    try:
                        if isinstance(patched_module, PatchedModule):
                            patched_module._restore_device_attribute_hierarchy()
                    except Exception as error:
                        first_error = first_error or error
                for patched_module in list(cls._patched_modules):
                    try:
                        if isinstance(patched_module, PatchedModule):
                            patched_module._update_state(PatchedModuleState.DETACHED)
                        patched_module._unpatch()
                    except Exception as error:
                        first_error = first_error or error
                if first_error is not None:
                    cls._cleanup_pending = True
                    raise first_error
                cls._intercepted_classes.clear()
                cls._cleanup_pending = False
        finally:
            if cls._original_module_init is not None:
                torch.nn.Module.__init__ = cls._original_module_init
            cls._torch_patched = False

    @classmethod
    def _is_allowed_to_tune(cls, module: torch.nn.Module) -> bool:
        """Check if the module is allowed to tune.

        If automatic patching is used, all modules, even those for internal torch use, will be intercepted.
        Those have to be rejected, otherwise JIT compilation will fail.

        Built-in defaults (``_DEFAULT_PATCH_EXCLUDE``) always apply;
        user-supplied ``jit_config.patch_exclude`` entries are additive.
        """
        patch_exclude = _DEFAULT_PATCH_EXCLUDE + config.patch_exclude
        module_cls = module.__class__
        if f"{module_cls.__module__}.{module_cls.__qualname__}" in patch_exclude:
            return False

        module_name = module_cls.__module__
        if any(module_name == p or module_name.startswith(p + ".") for p in patch_exclude):
            return False
        return True

    @classmethod
    def intercepted_classes(cls):
        """Get the intercepted classes."""
        return [f"{c.__module__}.{c.__name__}" for c in cls._intercepted_classes]


@contextmanager
def prepare_for_jit_tuning():
    """Context manager which prepares model for tuning.

    This context manager automatically intercepts creating torch modules and prepares them for tuning.
    The inference can happen outside of the context manager.

    Example:
        >>> with prepare_for_jit_tuning():
        ...     # torch is patched during this block
        ...     model = torch.nn.Linear(10, 5)
        ...     # torch is automatically unpatched when exiting the block
    """
    Patcher.patch_torch()
    try:
        yield
    finally:
        Patcher.unpatch_torch()


def patch_for_jit_tuning(func: Callable[..., T]) -> Callable[..., T]:
    """Wrapper that patches torch before function execution and unpatch_torch after.

    Args:
        func: The function to wrap with patching.

    Returns:
        Wrapped function that automatically patches/unpatches torch.

    Example:
        >>> @patch_for_jit_tuning
        ... def my_function():
        ...     # torch is patched during this function execution
        ...     model = torch.nn.Linear(10, 5)
        ...     return model
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        with prepare_for_jit_tuning():
            return func(*args, **kwargs)

    return wrapper


def jit_reset() -> None:
    """End the active JIT session and restore modules patched by AITune.

    Use this process-wide reset before switching JIT modes, beginning a new explicit
    registration generation, or explicitly tearing down a same-process JIT session.
    It is not required in the normal inference path. Existing
    :class:`~aitune.torch.jit.registration.JITRegistration` handles remain readable,
    but their modules report the detached state after reset.
    """
    try:
        Patcher._end_session_report()
    finally:
        Patcher.unpatch_torch(unpatch_modules=True)
        if Patcher._exit_handler is not None:
            atexit.unregister(Patcher._exit_handler)
        Patcher._exit_handler = None
        Patcher._session_mode = None
        Patcher._original_module_init = None
        PatchedModule.reset()
        InspectModule.reset()
