# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Torchao backend."""

import copy
import gc
import json
from collections.abc import Callable
from dataclasses import MISSING, dataclass, fields
from logging import getLogger
from pathlib import Path
from typing import Any, Literal, get_args

import torch
import torch.nn as nn
from dill import dumps, loads
from torchao.core.config import AOBaseConfig
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    PerTensor,
    quantize_,
)

try:
    from torchao.prototype.mx_formats.inference_workflow import (
        MXDynamicActivationMXWeightConfig,
        NVFP4DynamicActivationNVFP4WeightConfig,
    )
    from torchao.quantization.quantize_.common import KernelPreference

    MX_FORMATS_AVAILABLE = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)
except ImportError:
    MX_FORMATS_AVAILABLE = False


from aitune.torch.backend.backend import Backend, BackendConfig, BackendState, BuildMode
from aitune.torch.libs.torch_compile import TorchCompileMode, resolve_compile_dynamic
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import SampleStore
from aitune.utils.hashing import hash_string
from aitune.utils.serialization import json_serialize

logger = getLogger(__name__)

QuantizationType = Literal["int8wo", "int8dq", "fp8wo", "fp8dq", "mxfp8dq", "nvfp4dq"]
DEFAULT_QUANTIZATION = "fp8wo"

_HW_DEPENDENT_QUANTIZATIONS = frozenset({"nvfp4dq", "mxfp8dq"})

MXFP8DQ_BLOCK_SIZE_DIVISIBILITY = 32
NVFP4DQ_BLOCK_SIZE_DIVISIBILITY = 16

_BASE_QUANTIZATION_CONFIGS: dict = {
    "int8wo": Int8WeightOnlyConfig(),
    "int8dq": Int8DynamicActivationInt8WeightConfig(),
    "fp8wo": Float8WeightOnlyConfig(),
    "fp8dq": Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
}
if MX_FORMATS_AVAILABLE:
    _BASE_QUANTIZATION_CONFIGS["mxfp8dq"] = MXDynamicActivationMXWeightConfig(
        activation_dtype=torch.float8_e4m3fn,
        weight_dtype=torch.float8_e4m3fn,
        kernel_preference=KernelPreference.AUTO,
    )
    _BASE_QUANTIZATION_CONFIGS["nvfp4dq"] = NVFP4DynamicActivationNVFP4WeightConfig(
        use_dynamic_per_tensor_scale=True,
        use_triton_kernel=True,
    )


@dataclass
class TorchAOBackendConfig(BackendConfig):
    """Configuration for TorchAOBackend.

    Args:
        fullgraph: Passed to ``torch.compile`` to require a single graph.
        dynamic: Passed to ``torch.compile``. When ``None``, TorchAOBackend enables dynamic compilation only for
            graph specs with detected dynamic axes, without mutating this config.
        mode: Passed through to ``torch.compile(mode=...)``. Valid presets are ``"default"``,
            ``"reduce-overhead"``, ``"max-autotune"``, and ``"max-autotune-no-cudagraphs"``.
        quantization: Name of a built-in TorchAO quantization preset.
        quantization_config: Custom TorchAO quantization config. Use this instead of ``quantization`` for advanced
            TorchAO options.
        filter_fn: Optional TorchAO quantization predicate. It receives ``(module, fqn)`` and should return ``True``
            for modules that should be quantized. Compatibility preflight checks use the same predicate.
    """

    fullgraph: bool = False
    dynamic: bool | None = None
    mode: TorchCompileMode | None = "max-autotune"
    quantization: QuantizationType | None = None
    quantization_config: AOBaseConfig | None = None
    filter_fn: Callable[[nn.Module, str], bool] | None = None

    _QUANTIZATION_CONFIGS = _BASE_QUANTIZATION_CONFIGS

    def __post_init__(self):
        """Post init for TorchAOBackendConfig."""
        if self.quantization is not None and self.quantization_config is not None:
            raise ValueError("Only one of quantization or quantization_config should be provided.")
        if self.quantization is None and self.quantization_config is None:
            raise ValueError("Either quantization or quantization_config should be provided.")
        if self.mode is not None and self.mode not in get_args(TorchCompileMode):
            raise ValueError(f"Invalid mode: {self.mode!r}. Supported values: {get_args(TorchCompileMode)}")

        if not self.quantization_config:
            if not MX_FORMATS_AVAILABLE and self.quantization in _HW_DEPENDENT_QUANTIZATIONS:
                logger.debug(
                    "Hardware unavailable for %s — defer validation to _build(); _check_hardware_compatibility will raise",
                    self.quantization,
                )
                return

            self.quantization_config = self._get_quantization_config(self.quantization)

    def key(self) -> str:
        """Returns the key of the backend configuration."""
        config_dict = {
            "fullgraph": self.fullgraph,
            "dynamic": self.dynamic,
            "mode": self.mode,
        }
        if self.quantization_config is None:
            config_dict["quantization"] = self.quantization
        else:
            config_dict["quantization_config"] = self.quantization_config
        if self.filter_fn is not None:
            # The predicate is executable code, so hash its dill payload to include it in the cache key.
            config_dict["filter_fn"] = hash_string(dumps(self.filter_fn).hex())
        config_dict = json_serialize(config_dict)
        config_dict_str = json.dumps(config_dict)
        return hash_string(config_dict_str)

    def describe(self) -> str:
        """Returns the description of the backend."""
        if self.quantization_config is None:
            default = self.__class__(quantization=self.quantization)
        else:
            default = self.__class__(quantization_config=self.quantization_config)
        compile_options = self._get_changed_fields(
            self,
            default,
            exclude=["quantization", "quantization_config", "filter_fn"],
        )
        if self.quantization_config is None:
            quantization = f"quantization={self.quantization}"
            return ",".join([*compile_options, quantization])
        kwargs = {}
        for f in fields(self.quantization_config.__class__):
            if f.default is MISSING and f.default_factory is MISSING:
                kwargs[f.name] = getattr(self.quantization_config, f.name)

        changed_fields = self._get_changed_fields(
            self.quantization_config,
            self.quantization_config.__class__(*kwargs),
            include=list(kwargs.keys()),
        )
        quantization = f"quantization_config={self.quantization_config.__class__.__name__}({','.join(changed_fields)})"
        return ",".join([*compile_options, quantization])

    def to_dict(self):
        """Convert TorchAOBackendConfig to dict."""
        return {
            "fullgraph": self.fullgraph,
            "dynamic": self.dynamic,
            "mode": self.mode,
            "quantization_config": dumps(self.quantization_config),
            "filter_fn": dumps(self.filter_fn) if self.filter_fn is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TorchAOBackendConfig":
        """Initialise config from a plain dict (e.g. parsed from YAML).

        Supports two forms:

        - ``{"quantization": "int8wo"}`` — user-facing YAML form; ``quantization``
          must be one of the supported ``QuantizationType`` literals.
        - ``{"quantization_config": <bytes>}`` — internal checkpoint form produced
          by ``to_dict()``, where ``quantization_config`` is pickle-serialised.
        """
        data = dict(data)
        if isinstance(data.get("quantization_config"), bytes):
            data["quantization_config"] = loads(data["quantization_config"])
        if isinstance(data.get("filter_fn"), bytes):
            data["filter_fn"] = loads(data["filter_fn"])
        return cls(**data)

    def _get_quantization_config(self, quantization):
        if quantization not in self._QUANTIZATION_CONFIGS:
            valid_quantizations = list(self._QUANTIZATION_CONFIGS.keys())
            raise ValueError(f"Invalid quantization: {quantization}. Must be one of: {valid_quantizations}")
        return self._QUANTIZATION_CONFIGS[quantization]


class TorchAOBackend(Backend):
    """Backend that does torch quantization.

    If you would like to use customize quantization, you can pass in a quantization config.
    """

    _build_mode = BuildMode.JUST_IN_TIME

    # State dictionary keys
    STATE_TYPE = "type"
    STATE_CONFIG = "config"
    STATE_ORIG_MODULE = "orig_module"
    STATE_DATA = "data"
    STATE_SAMPLES = "samples"
    STATE_DEVICE = "device"
    STATE_COMPILE_DYNAMIC = "compile_dynamic"

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
        self._samples: SampleStore | None = None
        # Kept only for loading checkpoints written before samples became disk-backed.
        self._data = None
        self._compile_dynamic = self._config.dynamic

    def key(self) -> str:
        """Returns the key of the backend."""
        return f"{self.__class__.__name__}_{self._config.key()}"

    def describe(self) -> str:
        """Returns the description of the backend."""
        return f"{self.__class__.__name__}({self._config.describe()})"

    def _build(self, module: nn.Module, graph_spec: GraphSpec, samples: SampleStore, cache_dir: Path) -> Backend:
        """Builds the model with torchao quantization and torch.compile."""
        self._compile_dynamic = resolve_compile_dynamic(self._config.dynamic, graph_spec)

        self._save_config(cache_dir)

        self._orig_module = module
        self._samples = samples
        self._do_torchao_quantization()
        return self

    def _activate(self):
        """Activates backend."""
        self._do_torchao_quantization()

    def _infer(self, *args: Any, **kwargs: Any) -> Any:
        """Runs inference with the given arguments.

        Args:
            *args: inference arguments
            **kwargs: inference keyword arguments

        Returns:
            Any: The result of the inference.
        """
        with torch.no_grad():
            return self._quant_module(*args, **kwargs)

    def _deactivate(self):
        """Deactivates backend."""
        self._quant_module = None

    def _deploy(self):
        """Deploys the backend."""
        self._activate()
        self._samples = None
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
            self.STATE_SAMPLES: self._samples.to_dict() if self._samples is not None else None,
            self.STATE_DEVICE: self._device,
            self.STATE_COMPILE_DYNAMIC: self._compile_dynamic,
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
        samples_state = state_dict.get(cls.STATE_SAMPLES)
        backend._samples = SampleStore.from_dict(samples_state) if samples_state is not None else None
        backend._data = state_dict.get(cls.STATE_DATA)
        backend._device = state_dict[cls.STATE_DEVICE]
        backend._compile_dynamic = state_dict.get(cls.STATE_COMPILE_DYNAMIC, config.dynamic)
        backend._orig_module = module
        module.load_state_dict(state_dict[cls.STATE_ORIG_MODULE], strict=False)
        backend.state = BackendState.CHECKPOINT_LOADED

        return backend

    def _do_torchao_quantization(self):
        """Apply TorchAO quantization and warm up the compiled model."""
        if self._orig_module is None:
            raise RuntimeError("Backend has not been properly initialized. Please call build() first.")
        if self._samples is None and self._data is None:
            raise RuntimeError("Backend has no warmup data. Please call build() first.")

        self._check_hardware_compatibility(self._orig_module)

        model = copy.deepcopy(self._orig_module)
        self._orig_module.to("cpu")

        quantize_(model, config=self._config.quantization_config, device=self._device, filter_fn=self._config.filter_fn)
        self._quant_module = torch.compile(
            model=model,
            fullgraph=self._config.fullgraph,
            dynamic=self._compile_dynamic,
            mode=self._config.mode,
        )
        with torch.no_grad():
            for args, kwargs in self._iter_samples():
                self._quant_module(*args, **kwargs)

    def _iter_samples(self):
        """Iterate persisted samples, with support for legacy inline checkpoint data."""
        if self._samples is not None:
            return self._samples.iter_samples(self._device)
        return iter(self._data)

    def _check_hardware_compatibility(self, module: nn.Module) -> None:
        """Check hardware and model constraints for the selected quantization format.

        Args:
            module: The PyTorch module to validate.

        Raises:
            RuntimeError: If hardware or model constraints are not met.
        """
        self._check_nvfp4dq_compatibility(module)
        self._check_mxfp8dq_compatibility(module)

    def _iter_quantized_parameters(self, module: nn.Module):
        """Iterate parameters from modules selected by the TorchAO filter function.

        TorchAO applies ``filter_fn`` at the module level and skips modules where it returns ``False``. The
        compatibility preflight must mirror that behavior, otherwise it can reject parameters in layers that TorchAO
        would not quantize anyway, such as embeddings, output projections, or small ``Linear`` layers excluded by a
        model-specific predicate.
        """
        for module_name, submodule in module.named_modules():
            if self._config.filter_fn is not None and not self._config.filter_fn(submodule, module_name):
                continue
            for parameter_name, parameter in submodule.named_parameters(recurse=False):
                name = f"{module_name}.{parameter_name}" if module_name else parameter_name
                yield name, parameter

    def _check_nvfp4dq_compatibility(self, module: nn.Module) -> None:
        """Validate sm100+ GPU, bfloat16 dtype, and weight shape alignment for nvfp4dq.

        Pre-flight mirror of the assertions in
        ``torchao.prototype.mx_formats.inference_workflow._nvfp4_inference_linear_transform``
        (``is_sm_at_least_100()`` assert, bfloat16 assert, and ``weight.shape[-2] % 16`` check).
        The sm100+ and library requirement is captured by the module-level ``MX_FORMATS_AVAILABLE``
        flag (evaluated at import time).

        Args:
            module: The PyTorch module to validate.

        Raises:
            RuntimeError: If sm100+ GPU is unavailable, the model dtype is unsupported,
                or any weight's last two dimensions are not divisible by the block size.
        """
        is_nvfp4dq = self._config.quantization == "nvfp4dq" or (
            MX_FORMATS_AVAILABLE
            and isinstance(self._config.quantization_config, NVFP4DynamicActivationNVFP4WeightConfig)
        )
        if not is_nvfp4dq:
            return

        if not MX_FORMATS_AVAILABLE:
            raise RuntimeError(
                "nvfp4dq quantization requires sm100+ (Blackwell) GPU and TorchAO MX format support, "
                "which are not available on this hardware."
            )

        for name, param in self._iter_quantized_parameters(module):
            if param.dtype != torch.bfloat16:
                raise RuntimeError(f"nvfp4dq requires model dtype to be bfloat16; found {param.dtype}.")
            if param.ndim >= 2 and (
                param.shape[-1] % NVFP4DQ_BLOCK_SIZE_DIVISIBILITY != 0
                or param.shape[-2] % NVFP4DQ_BLOCK_SIZE_DIVISIBILITY != 0
            ):
                raise RuntimeError(
                    f"nvfp4dq requires the last two dimensions of every weight to be divisible by "
                    f"block_size {NVFP4DQ_BLOCK_SIZE_DIVISIBILITY}; {name} has shape {tuple(param.shape)}"
                )

    def _check_mxfp8dq_compatibility(self, module: nn.Module) -> None:
        """Validate sm100+ GPU, bfloat16 dtype, and block-size alignment for mxfp8dq.

        Pre-flight mirror of the bfloat16 assertion in
        ``torchao.prototype.mx_formats.inference_workflow._mx_inference_linear_transform``
        and the ``block_size=32`` field of ``MXDynamicActivationMXWeightConfig``.
        The sm100+ and library requirement is captured by the module-level ``MX_FORMATS_AVAILABLE``
        flag (evaluated at import time). Dtype is bfloat16-only —
        torchao asserts this explicitly; float32 is not accepted despite what older comments suggested.

        Args:
            module: The PyTorch module to validate.

        Raises:
            RuntimeError: If sm100+ GPU is unavailable, any parameter is not bfloat16,
                or a weight's last dimension is not divisible by MXFP8DQ_BLOCK_SIZE_DIVISIBILITY.
        """
        is_mxfp8dq = self._config.quantization == "mxfp8dq" or (
            MX_FORMATS_AVAILABLE and isinstance(self._config.quantization_config, MXDynamicActivationMXWeightConfig)
        )
        if not is_mxfp8dq:
            return

        if not MX_FORMATS_AVAILABLE:
            raise RuntimeError(
                "mxfp8dq quantization requires sm100+ (Blackwell) GPU and TorchAO MX format support, "
                "which are not available on this hardware."
            )

        for name, param in self._iter_quantized_parameters(module):
            if param.dtype != torch.bfloat16:
                raise RuntimeError(f"mxfp8dq requires model dtype to be bfloat16; found {param.dtype}.")
            if param.ndim >= 2 and param.shape[-1] % MXFP8DQ_BLOCK_SIZE_DIVISIBILITY != 0:
                raise RuntimeError(
                    f"mxfp8dq requires the last dimension of every weight to be divisible by "
                    f"block_size {MXFP8DQ_BLOCK_SIZE_DIVISIBILITY}; {name} has shape {tuple(param.shape)}"
                )
