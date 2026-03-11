# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inplace model registry."""

import gc
from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aitune.torch.module.wrapper_module import Module


class ModuleRegistry:
    """Registry for inplace modules."""

    def __init__(self) -> None:
        """Initialize ModuleRegistry."""
        self._registry: OrderedDict[str, Module] = OrderedDict()

    def register(self, name: str, module: "Module") -> None:
        """Register a module."""
        self._registry[name] = module

    def unregister(self, name: str) -> None:
        """Unregister a module."""
        if name in self._registry:
            del self._registry[name]

    def clear(self) -> None:
        """Removes already registered modules."""
        self._registry = OrderedDict()
        gc.collect()

    @property
    def modules(self) -> dict[str, "Module"]:
        """Get all registered modules."""
        return self._registry

    def get(self, name: str) -> "Module":
        """Get a module."""
        if name not in self._registry:
            raise ValueError(f"Module {name} not found in registry.")
        return self._registry[name]


MODULE_REGISTRY = ModuleRegistry()
