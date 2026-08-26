# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tuned module."""

from collections import OrderedDict
from typing import Any

import torch

from aitune.global_context import MODULE_CONTEXT_KEY, global_context
from aitune.torch.backend.backend import Backend
from aitune.torch.config import AITuneConfig
from aitune.torch.config import config as global_config
from aitune.torch.module.forward_signature import ForwardSignature
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.module.sample_store import Sample


class TunedModule:
    """Module that does inference using tuned backend."""

    # Dictionary keys
    BACKENDS_KEY = "backends"
    CHECK_GRAPH_KEY = "check_graph"
    IS_ANY_JIT_KEY = "is_any_jit"
    MODULE_NAME_KEY = "module_name"
    FORWARD_SIGNATURE_KEY = "forward_signature"
    TYPE_KEY = "type"

    # Error messages
    ERROR_NO_BACKENDS = "No backends provided"
    ERROR_UNKNOWN_BACKEND = "Unknown backend type: {}"
    ERROR_NO_BACKEND_FOUND = "No backend found for a graph with metadata: {}"

    def __init__(
        self,
        backends: OrderedDict[SampleMetadata, Backend],
        module_name: str,
        forward_signature: ForwardSignature,
        check_graph: bool = True,
        config: AITuneConfig | None = None,
    ) -> None:
        """Initializes module.

        Args:
            backends: dictionary of backends to be used for inference.
            module_name: Name of the module, used to tag hardware metrics during inference.
            forward_signature: Signature of the module's forward method.
            check_graph: whether to check the graph of the sample before calling the backend.
                If True, the graph of the sample is checked against the graph of the backend.
                If False, the graph of the sample is not checked against the graph of the backend.
            config: Configuration for the module, if not provided, global config is used.
        """
        self._backends = backends
        self._module_name = module_name
        self._check_graph = check_graph
        self._forward_signature = forward_signature
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
        forward_inputs = self._forward_signature.normalize(args, kwargs)
        sample = (forward_inputs.args, forward_inputs.kwargs)
        with global_context:
            global_context.set(MODULE_CONTEXT_KEY, self._module_name)
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
        forward_inputs = self._forward_signature.normalize(args, kwargs)
        sample_metadata = SampleMetadata.from_inputs(forward_inputs.arguments, strict=self._config.strict_mode)
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
            module = module.to(
                global_config.device_after_tuning
            )  # no module needed, we can move to meta(default) device

        for sample_metadata_data, backend_data in state_dict[TunedModule.BACKENDS_KEY]:
            if backend_data[TunedModule.TYPE_KEY] not in backend_classes:
                raise ValueError(TunedModule.ERROR_UNKNOWN_BACKEND.format(backend_data[TunedModule.TYPE_KEY]))

            backend_class = backend_classes[backend_data[TunedModule.TYPE_KEY]]
            backend = backend_class.from_dict(module, backend_data)
            backends[SampleMetadata.from_dict(sample_metadata_data)] = backend

        return TunedModule(
            backends=backends,
            check_graph=state_dict[TunedModule.CHECK_GRAPH_KEY],
            module_name=state_dict.get(TunedModule.MODULE_NAME_KEY, "Unknown"),  # for backward compatibility
            forward_signature=ForwardSignature.from_dict(state_dict[TunedModule.FORWARD_SIGNATURE_KEY]),
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
            self.MODULE_NAME_KEY: self._module_name,
            self.FORWARD_SIGNATURE_KEY: self._forward_signature.to_dict(),
        }
