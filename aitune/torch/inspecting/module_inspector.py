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

import time
from logging import getLogger
from typing import Any

import torch

from aitune.torch.inspecting.module_info import ModuleInfo
from aitune.torch.utils.cuda import synchronize

logger = getLogger(__name__)


class ModuleInspector:
    """Class for inspecting PyTorch modules and tracking their execution."""

    def __init__(self):
        """Initialize the module inspector."""
        self._module_info: dict[Any, ModuleInfo] = {}
        self._original_forward: dict[Any, Any] = {}
        self._inspected_objects: set = set()  # Track objects that have been inspected
        self._max_recursion_depth = 5

    def inspect(self, obj: Any) -> None:
        """Inspect an object and its members for PyTorch modules.

        Args:
            obj: The object to inspect
        """
        self.reset()
        self._inspect_object(obj)

    def reset(self) -> None:
        """Reset the inspector state."""
        # Restore original forward methods
        for module, original_forward in self._original_forward.items():
            module.forward = original_forward

        self._inspected_objects.clear()  # Clear the set when starting a new inspection

        self._module_info.clear()
        self._original_forward.clear()

    def get_modules(self) -> list[ModuleInfo]:
        """Get list of top-level executed modules or their first executed children.

        Returns:
            List of modules that were executed at the top level or their first executed children
            if the parent wasn't executed.
        """
        executed_modules = []

        def parent_called(module_info: ModuleInfo) -> bool:
            if module_info.parent is None:
                return False

            if module_info.parent.forward_called:
                return True

            return parent_called(module_info.parent)

        for module_info in self._module_info.values():
            if module_info.forward_called and not parent_called(module_info):
                executed_modules.append(module_info)

        return executed_modules

    def _inspect_object(self, obj: Any, depth: int = 1) -> None:
        """Inspect an object and its members for PyTorch modules.

        Args:
            obj: The object to inspect
            depth: The depth of the method recursion
        """
        # Check recursion depth
        if depth > self._max_recursion_depth:
            logger.debug("Maximum recursion depth (%d) reached, stopping inspection", self._max_recursion_depth)
            return

        # Skip Python built-in objects
        if obj.__class__.__module__ == "builtins":
            return

        # Skip if object has already been inspected
        obj_id = id(obj)
        if obj_id in self._inspected_objects:
            return

        self._inspected_objects.add(obj_id)

        # Check if object is a PyTorch module
        try:
            if isinstance(obj, torch.nn.Module):
                logger.debug("Inspecting module: %s", obj.__class__.__name__)
                self._inspect_module(obj)
                return
        except Exception:
            pass

        logger.debug("Inspecting object: %s", obj.__class__.__name__)
        self._inspect_members(obj, depth=depth)

    def _inspect_members(self, obj: Any, depth: int) -> None:
        """Inspect the members of an object.

        Args:
            obj: The object to inspect
            depth: The depth of the inspect object method recursion
        """
        members = {}
        for name in dir(obj):
            if name.startswith("__"):
                continue

            try:
                member = getattr(obj, name)
                if name not in members:
                    members[name] = member
            except Exception:
                continue

        for name, member in members.items():
            if not isinstance(member, torch.nn.Module):
                self._inspect_object(member, depth=depth + 1)
                continue

            try:
                if member in self._module_info:
                    continue

                logger.debug("Inspecting module: %s (%s)", name, member.__class__.__name__)
                self._inspect_module(member, name=name)
            except Exception:
                continue

    def _inspect_module(self, module, name: str | None = None) -> None:
        """Start inspecting a module and its submodules.

        Args:
            module: The PyTorch module to inspect
            name: The name of the module
        """
        self._register_module(module, name=name)

    def _register_module(self, module, parent: ModuleInfo | None = None, name: str | None = None) -> ModuleInfo | None:
        """Register a module and its submodules in the inspector.

        Args:
            module: The module to register
            parent: The parent module info if any
            name: The name of the module
        """
        if module in self._module_info:
            return None

        self._wrap_forward_methods(module)

        name = name or module.__class__.__name__

        module_info = ModuleInfo(name=name, module=module, parent=parent)
        self._module_info[module] = module_info

        for child_name, child in module.named_children():
            child_name = f"{name}.{child_name}" if name else child_name
            child_info = self._register_module(child, module_info, child_name)
            if child_info is not None:
                module_info.children.append(child_info)

        return module_info

    def _wrap_forward_methods(self, module) -> None:
        """Wrap forward methods of a module and its submodules.

        Args:
            module: The module whose forward method to wrap
        """
        if module in self._original_forward:
            return

        original_forward = module.forward
        self._original_forward[module] = original_forward

        def wrapped_forward(*args, **kwargs):
            module_info = self._module_info[module]
            module_info.forward_called = True
            module_info.execution_count += 1

            # Record input types
            input_info = {
                "args": [type(arg).__name__ for arg in args],
                "kwargs": {k: type(v).__name__ for k, v in kwargs.items()},
            }
            module_info.input_types.append(input_info)

            # Synchronize CUDA before starting timing
            synchronize()

            # Measure execution time
            start_time = time.perf_counter()
            output = original_forward(*args, **kwargs)

            # Synchronize CUDA after execution
            synchronize()

            # Collect end time
            end_time = time.perf_counter()

            # Update execution time
            module_info.total_execution_time += end_time - start_time

            # Record output type
            if isinstance(output, (tuple, list)):
                output_info = {"type": type(output).__name__, "elements": [type(x).__name__ for x in output]}
            else:
                output_info = {"type": type(output).__name__}
            module_info.output_types.append(output_info)

            return output

        module.forward = wrapped_forward
