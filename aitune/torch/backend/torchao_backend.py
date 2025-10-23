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
"""Torchao backend."""

import copy
import gc
import json
from dataclasses import MISSING, dataclass, fields
from logging import getLogger
from pathlib import Path
from typing import Any, Literal

import nvtx
import torch
import torch.nn as nn
from dill import dumps, loads
from torchao.core.config import AOBaseConfig
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8WeightOnlyConfig,
    FPXWeightOnlyConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    PerTensor,
    quantize_,
)

from aitune.torch.backend.backend import Backend, BackendConfig, BackendState
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.utils.hashing import hash_string

logger = getLogger(__name__)

QuantizationType = Literal["int4wo", "int8wo", "int8dq", "fp8wo", "fp8dq", "fp6e3m2", "fp5e2m2", "fp4e2m1"]
DEFAULT_QUANTIZATION = "fp8wo"


@dataclass
class TorchAOBackendConfig(BackendConfig):
    """Configuration for TorchAOBackend."""

    quantization: QuantizationType | None = None
    quantization_config: AOBaseConfig | None = None

    _QUANTIZATION_CONFIGS = {
        "int4wo": Int4WeightOnlyConfig(group_size=32),
        "int8wo": Int8WeightOnlyConfig(),
        "int8dq": Int8DynamicActivationInt8WeightConfig(),
        "fp8wo": Float8WeightOnlyConfig(),
        "fp8dq": Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
        "fp6e3m2": FPXWeightOnlyConfig(3, 2),
        "fp5e2m2": FPXWeightOnlyConfig(2, 2),
        "fp4e2m1": FPXWeightOnlyConfig(2, 1),
    }

    def __post_init__(self):
        """Post init for TorchAOBackendConfig."""
        if self.quantization is not None and self.quantization_config is not None:
            raise ValueError("Only one of quantization or quantization_config should be provided.")
        if self.quantization is None and self.quantization_config is None:
            raise ValueError("Either quantization or quantization_config should be provided.")

        if not self.quantization_config:
            self.quantization_config = self._get_quantization_config(self.quantization)

    def key(self) -> str:
        """Returns the key of the backend configuration."""
        config_dict = {
            "quantization_config": self.quantization_config,
        }
        config_dict = self._to_json(config_dict)
        config_dict_str = json.dumps(config_dict)
        key = hash_string(config_dict_str)
        return key

    def describe(self) -> str:
        """Returns the description of the backend."""
        kwargs = {}
        for f in fields(self.quantization_config.__class__):
            if f.default is MISSING and f.default_factory is MISSING:
                kwargs[f.name] = getattr(self.quantization_config, f.name)

        changed_fields = self._get_changed_fields(
            self.quantization_config,
            self.quantization_config.__class__(*kwargs),
            include=list(kwargs.keys()),
        )
        return f"quantization_config={self.quantization_config.__class__.__name__}({','.join(changed_fields)})"

    def to_dict(self):
        """Convert TorchAOBackendConfig to dict."""
        return {
            "quantization_config": dumps(self.quantization_config),
        }

    @classmethod
    def from_dict(cls, state_dict: dict):
        """Convert dict to TorchAOBackendConfig."""
        return cls(quantization_config=loads(state_dict["quantization_config"]))

    def _get_quantization_config(self, quantization):
        if quantization not in self._QUANTIZATION_CONFIGS:
            valid_quantizations = list(self._QUANTIZATION_CONFIGS.keys())
            raise ValueError(f"Invalid quantization: {quantization}. Must be one of: {valid_quantizations}")
        return self._QUANTIZATION_CONFIGS[quantization]


class TorchAOBackend(Backend):
    """Backend that does torch quantization.

    Supported quantizations:
        - int4wo
        - int8wo
        - int8dq
        - fp8wo
        - fp8dq
        - fp6e3m2
        - fp5e2m2
        - fp4e2m1

    If you would like to use customize quantization, you can pass in a quantization config.
    """

    is_jit = True

    # State dictionary keys
    STATE_TYPE = "type"
    STATE_CONFIG = "config"
    STATE_ORIG_MODULE = "orig_module"
    STATE_DATA = "data"
    STATE_DEVICE = "device"

    def __init__(
        self,
        config: TorchAOBackendConfig | None = None,
    ):
        """Initializes backend.

        Args:
            config: The configuration to use.
        """
        super().__init__()
        # initialize variables
        self._config = config or TorchAOBackendConfig(quantization=DEFAULT_QUANTIZATION)

        # build variables
        self._quant_module = None
        self._orig_module = None
        self._data = None

    def key(self) -> str:
        """Returns the key of the backend."""
        return f"{self.__class__.__name__}_{self._config.key()}"

    def describe(self) -> str:
        """Returns the description of the backend."""
        return f"{self.__class__.__name__}({self._config.describe()})"

    def _build(self, module: nn.Module, graph_spec: GraphSpec, data: list[Sample], cache_dir: Path) -> Backend:
        """Builds the model with torchao quantization and torch.compile."""
        self._save_config(cache_dir)

        self._orig_module = module
        self._data = data
        self._do_torchao_quantization()
        return self

    def _activate(self):
        """Activates backend."""
        self._do_torchao_quantization()

    @nvtx.annotate(message="TorchAOBackend.infer", domain="AITune", color="cyan")
    def _infer(self, *args: Any, **kwargs: Any) -> Any:
        """Runs inference with the given arguments.

        Args:
            *args: inference arguments
            **kwargs: inference keyword arguments

        Returns:
            Any: The result of the inference.
        """
        with torch.inference_mode():
            return self._quant_module(*args, **kwargs)

    def _deactivate(self):
        """Deactivates backend."""
        self._quant_module = None

    def _deploy(self):
        """Deploys the backend."""
        self._activate()
        self._data = None
        gc.collect()

    def _save_config(self, cache_dir: Path):
        """Store the backend configuration to a file."""
        self._config.to_json(cache_dir / "config.json")

    def to_dict(self):
        """Returns the state_dict of the backend."""
        if self._orig_module is None:
            raise RuntimeError("Backend has not been properly initialized. Please call build() first.")
        return {
            self.STATE_TYPE: self.__class__.__name__,
            self.STATE_CONFIG: self._config.to_dict(),
            self.STATE_ORIG_MODULE: self._orig_module.state_dict(),
            self.STATE_DATA: self._data,
            self.STATE_DEVICE: self._device,
        }

    @classmethod
    def from_dict(cls, module: torch.nn.Module | None, state_dict: dict):
        """Creates a backend from a state_dict."""
        if state_dict.get(cls.STATE_TYPE) != cls.__name__:
            raise ValueError(f"Invalid state_dict type: {state_dict.get(cls.STATE_TYPE)}")

        if module is None:
            raise ValueError("Module is required to create a backend from a state_dict.")

        config = TorchAOBackendConfig.from_dict(state_dict[cls.STATE_CONFIG])

        backend = cls(config=config)
        backend._data = state_dict[cls.STATE_DATA]
        backend._device = state_dict[cls.STATE_DEVICE]
        backend._orig_module = module
        module.load_state_dict(state_dict[cls.STATE_ORIG_MODULE], strict=False)
        backend.state = BackendState.CHECKPOINT_LOADED

        return backend

    def _do_torchao_quantization(self):
        model = copy.deepcopy(self._orig_module)
        self._orig_module.to("cpu")

        quantize_(model, config=self._config.quantization_config, device=self._device)
        self._quant_module = torch.compile(model=model, mode="max-autotune")
        with torch.inference_mode():
            for args, kwargs in self._data:
                self._quant_module(*args, **kwargs)
