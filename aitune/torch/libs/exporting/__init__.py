# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared tools for exporting modules through a flat tensor interface.

Models accept Python structures, while AITune uses ordered tensors as the compiled interface.
Recorded metadata defines the same order for example inputs, shape constraints, and outputs.
The adapter reconstructs the Python call inside the exported module, keeping backend code
independent of the original argument structure.

The adapter also exposes recorded tensor mutations as outputs. Backends use GraphSpec's forward effects
specification to apply those changes to current inputs and restore the Python result. See
docs/guides/advanced/execution_graphs.md for the flat interface design example.

Torch and ONNX export share this implementation so argument reconstruction, tensor ordering, and
shape rules stay aligned with inference. ONNX execution and source-model metadata live separately
in aitune.torch.libs.runtime and aitune.torch.libs.onnx_model_info; existing ONNX artifacts bypass export.
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
