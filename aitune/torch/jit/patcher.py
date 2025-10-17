# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module for inspecting PyTorch models and tracking their execution."""

import atexit
import inspect
import logging
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import TypeVar

import torch

from aitune.torch.jit.config import config
from aitune.torch.jit.inspect_module import InspectModule
from aitune.torch.jit.patched_module import PatchedModule

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
    _blacklisted_packages = {
        "torch.jit",
        "torch._inductor",
        "torch._dynamo",
        "torch.fx",
        "torch.export",
        "torch.export._trace",
        "torch.export._export",
    }

    @classmethod
    def patch_torch(cls):
        """Patch torch.nn.Module to track execution."""
        if cls._original_module_init is None:
            # this will be called only once
            cls._original_module_init = torch.nn.Module.__init__
            atexit.register(InspectModule.on_python_exit if config.inspect_mode else PatchedModule.on_python_exit)

        def _patched_init(module, *args, **kwargs):
            cls._original_module_init(module, *args, **kwargs)
            if cls._is_allowed_to_tune(module):
                cls._intercepted_classes.add(module.__class__)
                pm = InspectModule(module) if config.inspect_mode else PatchedModule(module)
                cls._patched_modules.append(pm)

        torch.nn.Module.__init__ = _patched_init

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
        module_info = inspect.getmodule(module.__class__)
        if module_info is not None and module_info.__package__ in cls._blacklisted_packages:
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
    InspectModule.reset() if config.inspect_mode else PatchedModule.reset()
