# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module for inspecting PyTorch models and tracking their execution."""

import atexit
import inspect
import logging
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import TypeVar

import torch

from aitune.torch.config import AITuneMode
from aitune.torch.jit.config import JITMode, config
from aitune.torch.jit.inspect_module import InspectModule
from aitune.torch.jit.patched_module import PatchedModule
from aitune.torch.tune_data.reporting import report_tune_run_start

T = TypeVar("T")


logger = logging.getLogger(__name__)


# Built-in exclusions the JIT patcher always applies. User-supplied entries on ``jit_config``
# (``extra_patch_exclude_packages`` / ``extra_patch_exclude_modules``) add to these — they cannot be
# removed via config to avoid breaking tuning by accident (e.g. dropping ``torch_tensorrt``
# leads to wrapt-decorated forwards in the compiled gm and crashes ``torch_tensorrt.save``).
_DEFAULT_PATCH_EXCLUDE_PACKAGES: tuple[str, ...] = (
    "torch.jit",
    "torch._inductor",
    "torch._dynamo",
    "torch.fx",
    "torch.export",
    "torch_tensorrt",
)

_DEFAULT_PATCH_EXCLUDE_MODULES: tuple[type[torch.nn.Module], ...] = (
    torch.nn.ModuleList,
    torch.nn.ModuleDict,
    torch.nn.Sequential,
    torch.nn.ParameterList,
    torch.nn.ParameterDict,
)


class Patcher:
    """Patcher class.

    Allows intercepting creating torch modules to register them with PatchedModule.
    """

    _patched_modules: list[PatchedModule | InspectModule] = []
    _intercepted_classes: set[type[torch.nn.Module]] = set()  # for tracking purposes
    _original_module_init: Callable | None = None

    @classmethod
    def patch_torch(cls):
        """Patch torch.nn.Module to track execution."""
        if cls._original_module_init is None:
            # this will be called only once
            cls._original_module_init = torch.nn.Module.__init__
            atexit.register(
                InspectModule.on_python_exit if config.mode == JITMode.INSPECT else PatchedModule.on_python_exit
            )

            if config.mode != JITMode.INSPECT:
                report_tune_run_start(AITuneMode.JIT)

        def _patched_init(module, *args, **kwargs):
            cls._original_module_init(module, *args, **kwargs)
            if cls._is_allowed_to_tune(module):
                cls._intercepted_classes.add(module.__class__)
                pm = InspectModule(module) if config.mode == JITMode.INSPECT else PatchedModule(module)
                cls._patched_modules.append(pm)

        torch.nn.Module.__init__ = _patched_init

    @classmethod
    def tune_deferred(cls):
        """Trigger tuning for all recorded modules in deferred mode.

        Call this after running at least one full forward pass through the pipeline so that every
        module has collected the samples it needs.  Intended for pipelines where modules are called
        a variable number of times per step (e.g. text-to-image or text-to-video), making it
        difficult to know when enough samples have been recorded inside the forward hook itself.
        """
        for module in list(PatchedModule.heads):
            try:
                module.try_tune()
            except Exception as e:
                logger.error("Failed to tune module %s: %s", module.__class__.__name__, e)

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
        if unpatch_modules:
            for patched_module in cls._patched_modules:
                patched_module._unpatch()
            cls._patched_modules.clear()
            cls._intercepted_classes.clear()
        if cls._original_module_init is not None:
            torch.nn.Module.__init__ = cls._original_module_init

    @classmethod
    def _is_allowed_to_tune(cls, module: torch.nn.Module) -> bool:
        """Check if the module is allowed to tune.

        If automatic patching is used, all modules, even those for internal torch use, will be intercepted.
        Those have to be rejected, otherwise JIT compilation will fail.

        Built-in defaults (``_DEFAULT_PATCH_EXCLUDE_MODULES``, ``_DEFAULT_PATCH_EXCLUDE_PACKAGES``)
        always apply; user-supplied ``jit_config.extra_patch_exclude_modules`` and
        ``extra_patch_exclude_packages`` are additive.
        """
        exclude_modules = _DEFAULT_PATCH_EXCLUDE_MODULES + config.extra_patch_exclude_modules
        if isinstance(module, exclude_modules):
            return False

        module_info = inspect.getmodule(module.__class__)
        if module_info is not None and module_info.__package__ is not None:
            pkg = module_info.__package__
            exclude_packages = _DEFAULT_PATCH_EXCLUDE_PACKAGES + config.extra_patch_exclude_packages
            if any(pkg == p or pkg.startswith(p + ".") for p in exclude_packages):
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


def jit_reset():
    """Reset the JIT patcher."""
    Patcher.unpatch_torch(unpatch_modules=True)
    atexit.unregister(PatchedModule.on_python_exit)
    atexit.unregister(InspectModule.on_python_exit)
    InspectModule.reset() if config.mode == JITMode.INSPECT else PatchedModule.reset()
