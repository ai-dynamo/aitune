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


class Patcher:
    """Patcher class.

    Allows intercepting creating torch modules to register them with PatchedModule.
    """

    _patched_modules: list[PatchedModule | InspectModule] = []
    _intercepted_classes: set[type[torch.nn.Module]] = set()  # for tracking purposes
    _original_module_init: Callable | None = None

    # Packages that should not be patched to avoid conflicts with JIT compilation
    _exclude_packages = {
        "torch.jit",
        "torch._inductor",
        "torch._dynamo",
        "torch.fx",
        "torch.export",
        "torch.export._trace",
        "torch.export._export",
    }

    # Container modules that should not be patched - they're organizational only, no computation
    # and can be created dynamically during forward passes via slicing
    _exclude_modules = (
        torch.nn.ModuleList,
        torch.nn.ModuleDict,
        torch.nn.Sequential,
        torch.nn.ParameterList,
        torch.nn.ParameterDict,
    )

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
        for module in cls._patched_modules:
            if not isinstance(module, PatchedModule):
                logger.warning("Module %s is not a PatchedModule, skipping", module.__class__.__name__)
                continue

            try:
                module.tune_deferred()
            except Exception as e:
                logger.error("Failed to tune module %s: %s", module.__class__.__name__, e)

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
        """
        # Skip container modules
        if isinstance(module, cls._exclude_modules):
            return False

        module_info = inspect.getmodule(module.__class__)
        if module_info is not None and module_info.__package__ in cls._exclude_packages:
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
