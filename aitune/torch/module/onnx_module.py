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
        self._ensure_session(device)

        inputs = prepare_onnx_inputs(self._session, args, kwargs)
        return run_onnx(self._session, inputs, device)

    def _ensure_session(self, device: torch.device) -> None:
        """Create a session when missing or when the input device changes."""
        if self._session is None or self._device != device:
            providers = (
                [("CUDAExecutionProvider", {"device_id": device.index or 0})]
                if device.type == "cuda"
                else ["CPUExecutionProvider"]
            )
            self._session = onnxruntime.InferenceSession(str(self.path), providers=providers)
            self._device = device

    def deactivate(self) -> None:
        """Release the runtime session."""
        self._session = None
