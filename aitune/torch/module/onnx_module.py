# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A runnable ONNX graph for AITune recording and tuning."""

from pathlib import Path

import onnxruntime
import torch
from torch import nn

from aitune.torch.libs.onnx.runtime import prepare_onnx_inputs, run_onnx


class OnnxModule(nn.Module):
    """Run an ONNX file with positional or named tensor inputs and named outputs.

    Wrap with ``aitune.torch.Module`` to record calls and select a backend with
    ``MaxThroughputStrategy``. Inputs must match the ONNX graph's names and dtypes.
    """

    def __init__(self, path: str | Path) -> None:
        """Keep the ONNX path and create a session lazily on the input device."""
        super().__init__()
        self.path = Path(path).resolve()
        self._session = None
        self._device = None

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
