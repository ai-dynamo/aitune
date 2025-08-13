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
"""Checkpoint abstract base class."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.nn import Module

from aitune.torch.module.wrapper_module import Module as WrapperModule


class Checkpoint(ABC):
    """Abstract base class for checkpoint operations.

    This class defines the interface for saving and loading tuned modules.
    """

    @abstractmethod
    def save(self, module: Module | WrapperModule, path: str | Path) -> None:
        """Save a module to a checkpoint.

        Args:
            module: The module to save. Can be a regular torch module or a wrapped module.
            path: The path where the module should be saved.

        Raises:
            NotImplementedError: If the concrete class does not implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def load(
        self,
        module: Module,
        path: str | Path,
        device_map: dict[str, str | torch.device] | None = None,
    ) -> Module | WrapperModule:
        """Load a module from a checkpoint.

        Args:
            module: The base module to load into.
            path: The path from where the module should be loaded.
            device_map: The device map to load module to.

        Returns:
            The loaded module, which may be either the original module type or a wrapped module.

        Raises:
            NotImplementedError: If the concrete class does not implement this method.
        """
        raise NotImplementedError
