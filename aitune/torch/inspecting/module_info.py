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
"""Module Info for inspected modules."""

from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class ModuleInfo:
    """Information about a PyTorch module."""

    name: str | None
    module: torch.nn.Module
    parent: Optional["ModuleInfo"] = None
    children: list["ModuleInfo"] = field(default_factory=list)
    forward_called: bool = False
    execution_count: int = 0
    total_execution_time: float = 0.0
    input_types: list[dict[str, Any]] = field(default_factory=list)
    output_types: list[dict[str, Any]] = field(default_factory=list)

    @property
    def module_type(self) -> type:
        """Get the type of the module."""
        return type(self.module)

    @property
    def average_execution_time(self) -> float:
        """Get average execution time of the forward function.

        Returns:
            Average execution time in seconds, or 0.0 if the module was never executed
        """
        return self.total_execution_time / self.execution_count if self.execution_count > 0 else 0.0

    @property
    def num_layers(self) -> int:
        """Get the number of layers in the module."""
        return len(list(self.module.named_children()))

    @property
    def num_parameters(self) -> int:
        """Get the number of parameters in the module."""
        return sum(p.numel() for p in self.module.parameters())

    @property
    def precisions(self) -> set:
        """Get the layers precisions of the module and return a set of unique precisions.

        Returns:
            Set of unique precisions
        """
        layer_precisions = set()
        for _, module in self.module.named_modules():
            if list(module.parameters()):  # Only check modules with parameters
                # Get the dtype of the first parameter (they should all be the same)
                param_dtype = next(module.parameters()).dtype
                layer_precisions.add(param_dtype)

        return layer_precisions


class InspectedModulesInfo:
    """Information about inspected modules."""

    def __init__(self, total_execution_time: float):
        """Initialize the inspected modules specification."""
        self._modules = {}
        self._total_execution_time = total_execution_time

    def add_module(self, module: ModuleInfo):
        """Add a module to the specification.

        Args:
            name: Name of the module.
            module: ModuleInfo object.
        """
        if module.name in self._modules:
            raise ValueError(f"Module with name {module.name} already exists")
        self._modules[module.name] = module

    def get_modules(
        self, min_execution_percentage: float | None = None, limit: int | None = None
    ) -> list["ModuleInfo"]:
        """Get the list of modules.

        Args:
            min_execution_percentage: Minimum execution percentage to include a module.
            limit: Maximum number of modules to return.

        Returns:
            List of ModuleInfo objects.
        """
        modules = []
        sorted_modules = sorted(
            self._modules.values(),
            key=lambda x: (-x.total_execution_time, x.name or ""),  # Negative for reverse=True, empty string for None
        )
        for module in sorted_modules:
            if min_execution_percentage is None or (
                module.total_execution_time / self._total_execution_time >= min_execution_percentage
            ):
                modules.append(module)

        if limit is not None:
            modules = modules[:limit]

        return modules

    def describe(self) -> None:
        """Describe the inspected modules specification."""
        print("Module Execution Summary:")  # noqa: T201
        print("=" * 138)  # noqa: T201
        print(  # noqa: T201
            f"{'Module Name':^20}  {'Calls':^8}  {'Total Time (s)':^15}  {'Avg Time (s)':^15}  {'% of Total':^10}  {'# of params':^15}  {'# of layers':^15}  {'precisions':^25}"
        )
        print("-" * 138)  # noqa: T201

        for info in self._modules.values():
            percentage = (
                (info.total_execution_time / self._total_execution_time) * 100 if self._total_execution_time > 0 else 0
            )

            precisions = ", ".join(str(p) for p in info.precisions)

            print(  # noqa: T201
                f"{info.name:<20}  "
                f"{info.execution_count:>8}  "
                f"{info.total_execution_time:>15.4f}  "
                f"{info.average_execution_time:>15.4f}  "
                f"{percentage:>10.2f}%  "
                f"{info.num_parameters:>15}  "
                f"{info.num_layers:>15}  "
                f"{precisions:>25}"
            )

        print("-" * 138)  # noqa: T201
        print(f"Total execution time: {self._total_execution_time:.6f} seconds")  # noqa: T201
        print("=" * 138)  # noqa: T201
