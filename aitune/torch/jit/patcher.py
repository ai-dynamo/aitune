# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module for inspecting PyTorch models and tracking their execution."""

import atexit
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from functools import wraps
from typing import TypeVar, cast

import torch

from aitune.torch.config import AITuneMode
from aitune.torch.jit.config import JITMode, config
from aitune.torch.jit.inspect_module import InspectModule
from aitune.torch.jit.patched_module import PatchedModule
from aitune.torch.tune_data.reporting import (
    has_active_report,
    report_inspection_details,
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
    _entry_path: str | None = None
    _mode: JITMode | None = None

    @staticmethod
    def _on_python_exit() -> None:
        """Dispatch exit diagnostics for the worker's chosen mode."""
        if Patcher._mode == JITMode.INSPECT:
            InspectModule.on_python_exit()
        else:
            PatchedModule.on_python_exit()

    @classmethod
    def _check_mode(cls) -> None:
        """Keep the mode fixed once the worker has installed its first wrapper."""
        if cls._mode is not None and cls._mode != config.mode:
            raise RuntimeError("JIT mode is fixed after module registration; start a fresh worker to change it")

    @classmethod
    def _register_module(
        cls,
        module: torch.nn.Module,
        *,
        explicit_head: bool = False,
        device_management: str = "aitune",
    ) -> PatchedModule | InspectModule:
        """Start ordinary reporting once and install one module wrapper."""
        if cls._mode is None:
            cls._mode = config.mode
            atexit.register(cls._on_python_exit)
            if config.mode != JITMode.INSPECT and not has_active_report():
                report_tune_run_start(AITuneMode.JIT)
        if config.mode == JITMode.INSPECT:
            patched_module: PatchedModule | InspectModule = InspectModule(module)
        else:
            patched_module = PatchedModule(module, explicit_head=explicit_head, device_management=device_management)
        cls._patched_modules.append(patched_module)
        return patched_module

    @classmethod
    def patch_torch(cls):
        """Intercept constructors for the worker's automatic JIT entry path."""
        if cls._entry_path == "explicit":
            raise RuntimeError("Automatic JIT discovery cannot follow explicit registration; start a fresh worker")
        cls._check_mode()
        cls._entry_path = "automatic"
        if cls._torch_patched:
            return

        if cls._original_module_init is None:
            cls._original_module_init = torch.nn.Module.__init__

        def _patched_init(module, *args, **kwargs):
            cls._original_module_init(module, *args, **kwargs)
            if cls._is_allowed_to_tune(module):
                cls._check_mode()
                cls._intercepted_classes.add(module.__class__)
                cls._register_module(module)

        torch.nn.Module.__init__ = _patched_init
        cls._torch_patched = True

    @classmethod
    def register_modules(
        cls, modules: Sequence[torch.nn.Module], *, device_management: str = "aitune"
    ) -> tuple[PatchedModule, ...]:
        """Preflight the complete batch, then install independent targets without rollback."""
        cls._validate_registration_targets(modules)
        if device_management == "external":
            cls._validate_external_candidates(modules)
        if config.mode == JITMode.INSPECT:
            raise RuntimeError("register_for_jit_tuning() does not support JITMode.INSPECT")
        if cls._entry_path is not None:
            raise RuntimeError(
                "Explicit JIT registration requires an unused worker; supply all targets once in a fresh worker"
            )

        cls._entry_path = "explicit"
        return cast(
            tuple[PatchedModule, ...],
            tuple(
                cls._register_module(module, explicit_head=True, device_management=device_management)
                for module in modules
            ),
        )

    @staticmethod
    def _validate_external_candidates(modules: Sequence[torch.nn.Module]) -> None:
        """Qualify the complete candidate set before touching an external source tree."""
        from aitune.torch.tune_strategy import (
            FirstWinsStrategy,
            LatencyBudgetStrategy,
            MaxThroughputStrategy,
            MinLatencyStrategy,
            OneBackendStrategy,
        )
        from aitune.torch.tune_strategy.multi_backend_strategy import MultiBackendStrategy
        from aitune.torch.utils.module import validate_external_backend

        for module in modules:
            strategy = config.resolve_strategy(module)
            if type(strategy) is OneBackendStrategy:
                backends = [strategy._backend]
            elif type(strategy) in (
                FirstWinsStrategy,
                MinLatencyStrategy,
                MaxThroughputStrategy,
                LatencyBudgetStrategy,
            ):
                backends = cast(MultiBackendStrategy, strategy)._backends
            else:
                raise ValueError("External device management requires a supported built-in strategy")
            if not backends:
                raise ValueError("External device management requires at least one backend candidate")
            for backend in backends:
                validate_external_backend(backend)
                backend._assert_supported_modules(module)

    @classmethod
    def _validate_registration_targets(cls, modules: Sequence[torch.nn.Module]) -> None:
        """Reject exclusions and overlapping ownership before installing any wrapper."""
        owned_ids: set[int] = set()
        for module in modules:
            if not cls._is_allowed_to_tune(module):
                raise ValueError(f"Explicit JIT target is excluded from tuning: {cls._module_label(module)}")
            tree_ids = {id(owned_module) for owned_module in module.modules()}
            if owned_ids & tree_ids:
                raise ValueError(
                    "Explicit JIT targets must have disjoint module ownership trees: " + cls._module_label(module)
                )
            owned_ids.update(tree_ids)

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
    def unpatch_torch(cls):
        """Restore module constructors while keeping existing JIT forwards installed."""
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
    owns_interception = not Patcher._torch_patched
    Patcher.patch_torch()
    try:
        yield
    finally:
        if owns_interception:
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
