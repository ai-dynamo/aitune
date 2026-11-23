# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Construct the YOLO ONNX source and a deterministic inference input."""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from aitune.torch.module import OnnxModule

ONNX_REVISION = "57657320425ee34056408a57ad9d29c4d4815bd8"
# Use supplied floating-point exports directly. The INT8 export is unsupported by TensorRT.
ONNX_VARIANTS = {
    "fp32": "model.onnx",
    "fp16": "model_fp16.onnx",
}


def onnx_model(variant: str = "fp16") -> OnnxModule:
    """Load the selected pre-exported YOLOv10n graph without converting its weights."""
    if variant not in ONNX_VARIANTS:
        raise ValueError(f"Unknown YOLO variant {variant!r}; choose from {', '.join(ONNX_VARIANTS)}")
    model_path = Path(
        hf_hub_download("onnx-community/yolov10n", f"onnx/{ONNX_VARIANTS[variant]}", revision=ONNX_REVISION)
    )
    return OnnxModule(model_path)


def sample_input(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Generate a reproducible RGB input for the graph's fixed 640x640 shape."""
    if dtype not in {torch.float16, torch.float32}:
        raise ValueError(f"Expected floating-point YOLO image input, got {dtype}")
    generator = torch.Generator(device="cuda").manual_seed(0)
    return torch.rand((1, 3, 640, 640), generator=generator, device="cuda", dtype=dtype)


def source_input(model: OnnxModule) -> torch.Tensor:
    """Read the input contract rather than inferring image precision from the export's filename."""
    info = model.graph_info
    if info.input_names != ["images"] or info.input_shapes.get("images") != [1, 3, 640, 640]:
        raise ValueError(f"Unexpected YOLO input contract: {info.input_names}, {info.input_shapes}")
    dtypes = {"FLOAT": torch.float32, "FLOAT16": torch.float16}
    declared_dtype = info.input_dtypes.get("images", "")
    if declared_dtype not in dtypes:
        raise ValueError(f"Unsupported YOLO image dtype in {model.path}: {declared_dtype}")
    return sample_input(dtypes[declared_dtype])
