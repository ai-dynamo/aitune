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
"""Tuned module."""

from collections import OrderedDict
from typing import Any

import torch

from aitune.torch.backend.backend import Backend
from aitune.torch.config import AITuneConfig
from aitune.torch.config import config as global_config
from aitune.torch.module.recording_module import Sample
from aitune.torch.module.sample_metadata import SampleMetadata


class TunedModule:
    """Module that does inference using tuned backend."""

    # Dictionary keys
    BACKENDS_KEY = "backends"
    CHECK_GRAPH_KEY = "check_graph"
    IS_ANY_JIT_KEY = "is_any_jit"
    TYPE_KEY = "type"

    # Error messages
    ERROR_NO_BACKENDS = "No backends provided"
    ERROR_UNKNOWN_BACKEND = "Unknown backend type: {}"
    ERROR_NO_BACKEND_FOUND = "No backend found for a graph with pytree metadata: {}"

    # Device names
    META_DEVICE = "meta"

    def __init__(
        self,
        backends: OrderedDict[SampleMetadata, Backend],
        check_graph: bool = True,
        config: AITuneConfig | None = None,
    ) -> None:
        """Initializes module.

        Args:
            module: module to be tuned.
            backends: dictionary of backends to be used for inference.
            check_graph: whether to check the graph of the sample before calling the backend.
                If True, the graph of the sample is checked against the graph of the backend.
                If False, the graph of the sample is not checked against the graph of the backend.
            config: Configuration for the module, if not provided, global config is used.
        """
        self._backends = backends
        self._check_graph = check_graph
        if len(self._backends) == 0:
            raise ValueError(self.ERROR_NO_BACKENDS)

        if check_graph or len(self._backends) > 1:
            self._backend_func = self.safe_call_backend
        else:
            self._unique_backend = list(self._backends.values())[0]
            self._backend_func = self.call_unique_backend

        self._config = config if config is not None else global_config

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the call through the tuned module."""
        sample = (args, kwargs)
        output = self._backend_func(sample)
        return output

    @property
    def device(self) -> torch.device:
        """Get the device of the module.

        Returns:
            The device the module is using.
        """
        return list(self._backends.values())[0].device

    @property
    def backends(self):
        """Get the backends of the module."""
        return self._backends

    def call_unique_backend(self, sample: Sample):
        """Calls the only backend without checking metadata."""
        args, kwargs = sample
        return self._unique_backend.infer(*args, **kwargs)

    def safe_call_backend(self, sample: Sample):
        """Calls the backend according to the metadata of the sample."""
        args, kwargs = sample
        sample_metadata = SampleMetadata.from_inputs(args, kwargs, strict=self._config.strict_mode)
        if sample_metadata not in self._backends:
            raise RuntimeError(self.ERROR_NO_BACKEND_FOUND.format(sample_metadata))

        backend = self._backends[sample_metadata]
        return backend.infer(*args, **kwargs)

    def activate(self):
        """Activates the module backends."""
        for backend in self._backends.values():
            backend.activate()

    def deactivate(self):
        """Deactivates the module backends."""
        for backend in self._backends.values():
            backend.deactivate()

    def deploy(self, device: torch.device | None):
        """Deploys the module backends."""
        for backend in self._backends.values():
            backend.deploy(device=device)

    @staticmethod
    def from_dict(module: torch.nn.Module, state_dict: dict):
        """Creates a TunedModule from a state_dict."""
        backend_classes = {cls.__name__: cls for cls in Backend.__subclasses__()}
        backends = OrderedDict()

        if not state_dict[TunedModule.IS_ANY_JIT_KEY]:
            module = module.to(TunedModule.META_DEVICE)  # no module needed, we can move to meta device

        for sample_metadata_data, backend_data in state_dict[TunedModule.BACKENDS_KEY]:
            if backend_data[TunedModule.TYPE_KEY] not in backend_classes:
                raise ValueError(TunedModule.ERROR_UNKNOWN_BACKEND.format(backend_data[TunedModule.TYPE_KEY]))

            backend_class = backend_classes[backend_data[TunedModule.TYPE_KEY]]
            backend = backend_class.from_dict(module, backend_data)
            backends[SampleMetadata.from_dict(sample_metadata_data)] = backend

        return TunedModule(
            backends=backends,
            check_graph=state_dict[TunedModule.CHECK_GRAPH_KEY],
        )

    def to_dict(self):
        """Returns the state_dict of the module."""
        backends_data = []
        is_any_jit = False
        for sample_metadata, backend in self._backends.items():
            backends_data.append((sample_metadata.to_dict(), backend.to_dict()))  # pytype: disable=attribute-error
            if backend.is_jit:  # pytype: disable=attribute-error
                is_any_jit = True

        return {
            self.BACKENDS_KEY: backends_data,
            self.CHECK_GRAPH_KEY: self._check_graph,
            self.IS_ANY_JIT_KEY: is_any_jit,
        }
