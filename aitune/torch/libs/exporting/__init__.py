# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared tools for exporting modules through a flat tensor interface.

Models accept Python structures, while AITune uses ordered tensors as the compiled interface.
Recorded metadata defines the same order for example inputs, shape constraints, and outputs.
The adapter reconstructs the Python call inside the exported module, keeping backend code
independent of the original argument structure.

These are export building blocks. Backend integration and restoration of caller-visible input
changes are added separately. See docs/guides/advanced/execution_graphs.md for the design example.
"""

from aitune.torch.libs.exporting.onnx_exporter import ONNXExporter
from aitune.torch.libs.exporting.preparation import TensorExportPreparation, prepare_tensor_export
from aitune.torch.libs.exporting.tensor_export_adapter import TensorExportAdapter
from aitune.torch.libs.exporting.torch_exporter import TorchExporter, TorchExportResult

__all__ = [
    "ONNXExporter",
    "TensorExportAdapter",
    "TensorExportPreparation",
    "TorchExporter",
    "TorchExportResult",
    "prepare_tensor_export",
]
