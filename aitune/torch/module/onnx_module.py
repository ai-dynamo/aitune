# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A runnable ONNX graph for AITune recording and tuning."""

from pathlib import Path

import onnxruntime
import torch
from torch import nn

from aitune.torch.libs.onnx.runtime import prepare_onnx_inputs, run_onnx
from aitune.torch.module.graph_spec import GraphSpec


class OnnxModule(nn.Module):
    """Run an ONNX file with positional or named tensor inputs and named outputs.

    Wrap with ``aitune.torch.Module`` to record calls and select a backend with
    ``MaxThroughputStrategy``. Inputs must match the ONNX graph's names and dtypes.
    Use ``for_checkpoint()`` with ``aitune.torch.load`` when the checkpoint
    contains a self-contained AOT backend and the ONNX file is not needed.
    """

    def __init__(self, path: str | Path | None = None, *, _checkpoint: bool = False) -> None:
        """Keep the ONNX path and create a session lazily on the input device."""
        if path is None and not _checkpoint:
            raise TypeError("An ONNX file is required; use OnnxModule.for_checkpoint() to load an AOT checkpoint")
        if path is not None and _checkpoint:
            raise ValueError("A checkpoint placeholder cannot have an ONNX file")
        super().__init__()
        self._path = Path(path).resolve() if path is not None else None
        self._input_names: tuple[str, ...] = ()
        self._output_names: tuple[str, ...] = ()
        self._session = None
        self._device = None

    @classmethod
    def for_checkpoint(cls) -> "OnnxModule":
        """Create a placeholder for loading a self-contained AOT checkpoint."""
        return cls(_checkpoint=True)

    @property
    def path(self) -> Path:
        """Return the ONNX source path, required for direct inference and tuning."""
        if self._path is None:
            raise RuntimeError("OnnxModule has no ONNX file; pass a path for direct inference or tuning")
        return self._path

    def preserve_tensor_names(self, graph_spec: GraphSpec) -> None:
        """Attach original ONNX names to recorded tensors without altering their observed shapes."""
        input_specs = [graph_spec.input_spec]
        if graph_spec.post_input_spec is not None:
            input_specs.append(graph_spec.post_input_spec)
        for metadata in input_specs:
            for locator, spec in metadata.tensor_data:
                key = locator.leaf_name
                spec.name = self._input_names[key] if isinstance(key, int) else key
                if spec.name not in self._input_names:
                    raise ValueError(f"Unknown ONNX input {spec.name!r}")
        for locator, spec in graph_spec.output_spec.tensor_data:
            spec.name = str(locator.leaf_name)
            if spec.name not in self._output_names:
                raise ValueError(f"Unknown ONNX output {spec.name!r}")

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Execute the graph using the same runtime as ONNXRuntimeBackend."""
        values = (*args, *kwargs.values())
        if not values:
            raise ValueError("OnnxModule requires at least one tensor input")

        device = values[0].device
        assert all(value.device == device for value in values), "OnnxModule inputs must share a device"
        assert self._device is None or self._device == device, (
            f"OnnxModule is on {self._device}, but input is on {device}; call offload() to switch devices"
        )
        self._ensure_session(device)

        inputs = prepare_onnx_inputs(self._session, args, kwargs)
        return run_onnx(self._session, inputs, device)

    @staticmethod
    def _providers(device: torch.device) -> list:
        """Return execution providers for the selected device."""
        if device.type == "cuda":
            return [("CUDAExecutionProvider", {"device_id": device.index})]
        return ["CPUExecutionProvider"]

    def _ensure_session(self, device: torch.device) -> None:
        """Create a session lazily without switching an existing session's device."""
        if self._session is None:
            self._device = device
            self._session = onnxruntime.InferenceSession(str(self.path), providers=self._providers(device))
            self._input_names = tuple(node.name for node in self._session.get_inputs())
            self._output_names = tuple(node.name for node in self._session.get_outputs())

    def offload(self, device: str | torch.device = "cpu") -> None:
        """Switch execution providers, using CPU for meta since ORT has no meta provider."""
        device = torch.device(device)
        if device.type == "meta":
            device = torch.device("cpu")

        if device.type not in {"cpu", "cuda"}:
            raise ValueError(f"OnnxModule does not support device {device}")

        # FIXME(kn): onnxruntime: Failed to parse value: "0"?
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        if self._session is not None and self._device != device:
            self._session.set_providers(self._providers(device))

        self._device = device

    def deactivate(self) -> None:
        """Release the runtime session."""
        self._session = None
        self._device = None
