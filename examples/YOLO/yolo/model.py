# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Construct the YOLO ONNX source and a deterministic inference input."""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from aitune.torch.module import OnnxModule

ONNX_REVISION = "57657320425ee34056408a57ad9d29c4d4815bd8"


def onnx_model() -> OnnxModule:
    """Load the pinned YOLOv10n ONNX graph."""
    model_path = Path(hf_hub_download("onnx-community/yolov10n", "onnx/model.onnx", revision=ONNX_REVISION))
    return OnnxModule(model_path)


def sample_input() -> torch.Tensor:
    """Generate a reproducible RGB input for the graph's fixed 640x640 shape."""
    generator = torch.Generator().manual_seed(0)
    return torch.rand((1, 3, 640, 640), generator=generator).cuda()
