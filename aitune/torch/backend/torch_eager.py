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

"""Torch eager backend."""

from pathlib import Path
from typing import Any

import nvtx
import torch
import torch.nn as nn

from aitune.torch.backend.backend import Backend, BackendState
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample


class TorchEagerBackend(Backend):
    """Backend that runs the model in eager mode without any optimizations."""

    # State dictionary keys
    STATE_TYPE = "type"
    STATE_ORIG_MODULE = "orig_module"
    STATE_GRAPH_SPEC = "graph_spec"
    STATE_DEVICE = "device"

    def __init__(self):
        """Initializes backend."""
        super().__init__()
        self._orig_module = None
        self._graph_spec = None

    def is_jit(self) -> bool:
        """Returns True if the backend is a JIT backend."""
        return True

    def describe(self) -> str:
        """Returns the description of the backend."""
        return f"{self.__class__.__name__}()"

    def _build(self, module: nn.Module, graph_spec: GraphSpec, data: list[Sample], cache_dir: Path) -> Backend:
        """Builds the model."""
        module.to(self._device)

        self._orig_module = module
        self._graph_spec = graph_spec

        # must be activated before returning the backend
        self._activate()

        return self

    def _activate(self):
        """Activates runner."""
        pass

    @nvtx.annotate(name="TorchEagerBackend.infer", domain="AITune", color="blue")
    def _infer(self, *args: Any, **kwargs: Any) -> Any:
        """Runs inference with the given arguments.

        Args:
            *args: inference arguments
            **kwargs: inference keyword arguments

        Returns:
            Any: The result of the inference.
        """
        with torch.inference_mode():
            return self._orig_module(*args, **kwargs)

    def _deactivate(self):
        """Deactivates runner."""
        pass

    def _deploy(self):
        """Deploys the backend."""
        pass

    def to_dict(self):
        """Returns the state_dict of the backend."""
        if self._orig_module is None:
            raise RuntimeError("Backend has not been properly initialized. Please call build() first.")
        return {
            self.STATE_TYPE: self.__class__.__name__,
            self.STATE_ORIG_MODULE: self._orig_module.state_dict(),
            self.STATE_GRAPH_SPEC: self._graph_spec,
            self.STATE_DEVICE: self.device,
        }

    @classmethod
    def from_dict(cls, module: torch.nn.Module | None, state_dict: dict):
        """Creates a backend from a state_dict."""
        if state_dict.get(cls.STATE_TYPE) != cls.__name__:
            raise ValueError(f"Invalid state_dict type: {state_dict.get(cls.STATE_TYPE)}")

        if module is None:
            raise ValueError("Module is required to create a backend from a state_dict.")

        backend = cls()
        backend._orig_module = module
        backend._graph_spec = state_dict[cls.STATE_GRAPH_SPEC]
        backend._set_device(state_dict[cls.STATE_DEVICE])
        module.load_state_dict(state_dict[cls.STATE_ORIG_MODULE], strict=False)
        backend.state = BackendState.CHECKPOINT_LOADED
        return backend
