# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Construct the Torch or ONNX source model for the BERT example."""

from pathlib import Path

import torch
from transformers import AutoModel

from aitune.torch.module import OnnxModule


def torch_model(model_name: str) -> torch.nn.Module:
    """Load the same model and output contract as the ONNX functional test."""
    return AutoModel.from_pretrained(model_name, attn_implementation="eager", return_dict=False).eval()


def onnx_model(path: Path, tokens: torch.Tensor, model_name: str) -> OnnxModule:
    """Export the selected pretrained model with the example's input/output contract."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        torch_model(model_name),
        (tokens[:1],),
        path,
        input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
            "pooler_output": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
        external_data=False,
    )
    return OnnxModule(path)
