# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Torch Inductor AOT backend."""

from copy import deepcopy
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any

import nvtx
import torch
import torch.nn as nn

from aitune.torch.backend.backend import Backend, BackendConfig, BackendState
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.utils.module import get_forward_arguments_names, offload
from aitune.torch.utils.shapes import (
    create_inputs_mapping,
    create_ordered_dynamic_shapes,
    print_dynamic_shapes,
    war_for_positional_arguments,
)

logger = getLogger(__name__)


@dataclass
class TorchInductorAotBackendConfig(BackendConfig):
    """Configuration for TorchInductorAotBackend.

    Args:
        inductor_configs (dict): Mapping of ``torch._inductor.config`` attribute names to values,
            passed to ``torch._inductor.aoti_compile_and_package()``.
            Call ``torch._inductor.list_options()`` to see all available keys.
            Example: ``{"max_autotune": True, "coordinate_descent_tuning": True}``
    """

    inductor_configs: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, state_dict: dict):
        """Convert dict to TorchInductorAotBackendConfig."""
        return cls(**state_dict)


class TorchInductorAotBackend(Backend):
    """Backend that compiles models using AOT Inductor (``torch._inductor.aoti_compile_and_package``).

    Requires PyTorch >= 2.6.

    Performs Ahead-of-Time compilation to a portable ``.pt2`` artifact saved to disk.
    Dynamic batch shapes are inferred automatically from ``graph_spec`` when a batch axis is
    detected. At inference time the artifact is loaded via ``torch._inductor.aoti_load_package``,
    enabling deployment without Python-interpreter overhead.

    Workflow::

        backend = TorchInductorAotBackend()
        # build / tune as usual through ait.tune()
        ait.save(model, "model.ait")
        # later
        ait.load(model, "model.ait")
    """

    is_jit = False

    # State dictionary keys
    STATE_TYPE = "type"
    STATE_COMPILED_MODEL_PATH = "compiled_model_path"
    STATE_DEVICE = "device"

    def __init__(self, config: TorchInductorAotBackendConfig | None = None):
        """Initialize TorchInductorAotBackend.

        Args:
            config: Configuration for AOT Inductor compilation.
        """
        super().__init__()
        self._config = config or TorchInductorAotBackendConfig()
        self._compiled_model_path: Path | None = None
        self._runner = None

    def key(self) -> str:
        """Returns the key of the backend."""
        return f"{self.__class__.__name__}_{self._config.key()}"

    def describe(self) -> str:
        """Returns the description of the backend."""
        return f"{self.__class__.__name__}({self._config.describe()})"

    def _build(self, module: nn.Module, graph_spec: GraphSpec, data: list[Sample], cache_dir: Path) -> Backend:
        """Export and compile the model with AOT Inductor, then load the runner."""
        self._save_config(cache_dir)

        module = module.eval().to(self._device)
        args, kwargs = deepcopy(data[0])
        args = tuple(a.to(self._device) if isinstance(a, torch.Tensor) else a for a in args)
        kwargs = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

        # torch.export.export specialises batch=1 as a constant; use batch>=2 when a batch axis
        # is present to keep the dimension symbolic (same pattern as TorchTensorRTAotBackend).
        if graph_spec.input_spec.has_batch_axis():
            max_batch_size = graph_spec.get_max_batch_size()
            batch_size = min(max_batch_size, 2)
            args, kwargs = graph_spec.input_spec.make_batch(args, kwargs, batch_size=batch_size)

        forward_args = get_forward_arguments_names(module.forward)
        dynamic_shapes = self._build_dynamic_shapes(args, kwargs, graph_spec, forward_args)

        if dynamic_shapes is not None:
            print_dynamic_shapes(dynamic_shapes)

        logger.info("Exporting model with torch.export.export.")
        with torch.no_grad():
            exported = torch.export.export(
                module,
                args,
                kwargs=kwargs if kwargs else None,
                dynamic_shapes=dynamic_shapes,
            )

        self._compiled_model_path = cache_dir / "model.pt2"
        logger.info("Compiling model with AOT Inductor to %s.", self._compiled_model_path)
        torch._inductor.aoti_compile_and_package(
            exported,
            package_path=str(self._compiled_model_path),
            inductor_configs=self._config.inductor_configs or {},
        )
        logger.info("AOT Inductor compilation complete with package path %s.", self._compiled_model_path)

        # The compiled artifact is self-contained; offload the original module to CPU
        # to free GPU memory now that the runner is loaded.
        offload(module, device="cpu")

        self._activate()
        return self

    def _build_dynamic_shapes(
        self, args: tuple, kwargs: dict, graph_spec: GraphSpec, forward_args: tuple[list[str], list[str]]
    ) -> dict | None:
        """Build a dynamic_shapes dict for ``torch.export.export``, keyed by parameter name.

        Creates one ``torch.export.Dim`` per symbolic dimension class:

        - **Batch axes** (``"batch*"``): a single base ``batch`` Dim whose min/max represent the
          base batch size. Axes with a multiplier > 1 use a derived expression
          (e.g. ``2 * batch_dim``), which requires PyTorch >= 2.4.
        - **Dynamic axes** (``"dim*"``): one ``Dim.AUTO`` per unique symbolic name (e.g. ``"dim1"``
          for sequence length); torch.export infers valid ranges and divisibility constraints.

        Returns:
            A dict mapping each forward parameter name to its ``{axis: Dim}`` constraints,
            covering both positional args and keyword arguments, or ``None`` if no dynamic
            dimension is found.
        """
        input_spec = graph_spec.input_spec
        if not any(ts.has_batch_axis() or ts.has_dynamic_axis() for ts in input_spec.tensor_specs):
            return None

        batch_dim = self._make_batch_dim(input_spec.tensor_specs)
        dynamic_dims = self._make_dynamic_dims(input_spec.tensor_specs)

        # Build flat map: tensor_spec_name → {axis: Dim} for all tensor specs
        spec_to_dims: dict[str, dict] = {}
        for tensor_spec in input_spec.tensor_specs:
            if tensor_spec.has_batch_axis() or tensor_spec.has_dynamic_axis():
                spec_to_dims[tensor_spec.name] = self._axis_dims_for_tensor(tensor_spec, batch_dim, dynamic_dims)
            else:
                spec_to_dims[tensor_spec.name] = {}

        # Map spec names to forward parameter names (handles both args and kwargs)
        fwd_args, fwd_kwargs = forward_args
        input_args, input_kwargs = create_inputs_mapping(input_spec)
        war_for_positional_arguments(input_args, fwd_args, fwd_kwargs)
        result = create_ordered_dynamic_shapes(fwd_args, fwd_kwargs, input_args, input_kwargs, spec_to_dims)

        # WAR: Non-tensor kwargs have to be mentioned in dynamic shapes - adding missing ones.
        # torch.export.export requires that every key present in the kwargs passed to it
        # appears in dynamic_shapes. Optional kwargs that are None at recording time are
        # absent from input_kwargs but are still forwarded to export.
        for key in kwargs:
            if key not in result:
                result[key] = {} if isinstance(kwargs[key], (dict, list)) else None

        return result

    @staticmethod
    def _make_batch_dim(tensor_specs) -> Any:
        """Create a ``torch.export.Dim`` for batch axes, or ``None`` if the batch is static.

        Uses the actual axis values from ``TensorSpec.min_shape`` / ``max_shape`` directly.
        For a CFG-doubled UNet with ``batch_sizes=[1, 2]``, ``axis_0 ∈ [2, 4]``, so the
        Dim range is ``[2, 4]`` and the axis constraint is just ``batch_dim`` with no multiplier.

        Returns ``None`` when ``min_val == max_val``: all recordings had the same batch axis
        size, so the dimension is static and should not be marked dynamic.
        """
        if not any(ts.has_batch_axis() for ts in tensor_specs):
            return None
        min_val = float("inf")
        max_val = 0
        for ts in tensor_specs:
            for axis in ts.get_batch_axis_multipliers():
                min_val = min(min_val, ts.min_shape[axis])
                max_val = max(max_val, ts.max_shape[axis])
        if min_val == max_val:
            return None
        return torch.export.Dim("batch", min=int(min_val), max=int(max_val))

    @staticmethod
    def _make_dynamic_dims(tensor_specs) -> dict[str, Any]:
        """Create one ``torch.export.Dim.AUTO`` per unique ``"dim*"`` symbolic name.

        ``Dim.AUTO`` lets torch.export infer the valid range and any divisibility
        constraints automatically from the model, avoiding false constraint-violation
        errors that arise when an explicit min/max range contains values that violate
        internal model guards (e.g. strided-convolution divisibility requirements).
        """
        names: set[str] = set()
        for ts in tensor_specs:
            for entry in ts.shape:
                if isinstance(entry, str) and entry.startswith("dim"):
                    names.add(entry)
        return dict.fromkeys(names, torch.export.Dim.AUTO)

    @staticmethod
    def _axis_dims_for_tensor(tensor_spec, batch_dim: Any, dynamic_dims: dict[str, Any]) -> dict[int, Any]:
        """Return the ``{axis_index: Dim}`` mapping for a single tensor spec."""
        axis_dims: dict[int, Any] = {}
        if batch_dim is not None:
            for axis in tensor_spec.get_batch_axis_multipliers():
                axis_dims[axis] = batch_dim
        for i, entry in enumerate(tensor_spec.shape):
            if isinstance(entry, str) and entry.startswith("dim"):
                axis_dims[i] = dynamic_dims[entry]
        return axis_dims

    def _activate(self):
        """Load the compiled model from disk."""
        logger.debug("Loading compiled AOT Inductor runner from %s.", self._compiled_model_path)
        device_index = self._device.index if self._device.index is not None else 0
        self._runner = torch._inductor.aoti_load_package(str(self._compiled_model_path), device_index=device_index)

    @nvtx.annotate(message="TorchInductorAotBackend.infer", domain="AITune", color="orange")
    def _infer(self, *args: Any, **kwargs: Any) -> Any:
        """Run inference with the compiled AOT Inductor runner.

        Args:
            *args: Inference arguments.
            **kwargs: Inference keyword arguments.

        Returns:
            Any: The result of the inference.
        """
        with torch.no_grad():
            return self._runner(*args, **kwargs)

    def _deactivate(self):
        """Deactivate backend."""
        self._runner = None

    def _deploy(self):
        """Deploy backend."""
        self._activate()

    def _save_config(self, cache_dir: Path):
        """Store the backend configuration to a file."""
        config_path = cache_dir / "config.json"
        self._config.to_json(config_path)
        logger.info("Config saved to %s", config_path)

    def to_dict(self) -> dict:
        """Returns the state_dict of the backend."""
        if self._compiled_model_path is None:
            raise RuntimeError("Backend has not been built yet. Please call build() first.")
        return {
            self.STATE_TYPE: self.__class__.__name__,
            self.STATE_COMPILED_MODEL_PATH: self._compiled_model_path,
            self.STATE_DEVICE: self._device,
        }

    @classmethod
    def from_dict(cls, module: nn.Module | None, state_dict: dict) -> "TorchInductorAotBackend":
        """Creates a backend from a state_dict."""
        if state_dict.get(cls.STATE_TYPE) != cls.__name__:
            raise ValueError(f"Invalid state_dict type: {state_dict.get(cls.STATE_TYPE)}")

        backend = cls()
        backend._compiled_model_path = state_dict[cls.STATE_COMPILED_MODEL_PATH]
        backend._device = state_dict[cls.STATE_DEVICE]
        backend.state = BackendState.CHECKPOINT_LOADED
        return backend
