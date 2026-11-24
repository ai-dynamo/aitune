# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate a saved YOLO checkpoint through Python inference."""

import argparse
from pathlib import Path
from typing import cast

import torch

from aitune.torch import Module, load
from aitune.torch.module import onnx_checkpoint_placeholder
from yolo.cmd_args import add_tuned_model_path_arg, add_variant_arg
from yolo.model import onnx_model, source_input


def run_inference(checkpoint: Path, variant: str = "fp16") -> None:
    """Check the restored checkpoint's input contract and finite detection outputs."""
    source = onnx_model(variant)
    images = source_input(source)
    tuned_model = cast(Module, load(onnx_checkpoint_placeholder(), checkpoint))
    try:
        artifact = tuned_model.artifact()
        if len(artifact.inputs) != 1 or artifact.inputs[0].dtype.value != str(images.dtype).removeprefix("torch."):
            raise RuntimeError(f"YOLO checkpoint input dtype does not match the {variant} source export")
        artifact_input_name = artifact.input_names[0]
        if artifact_input_name not in {"images", "input_x"}:
            raise RuntimeError(f"Unexpected YOLO input name: {artifact_input_name}")
        input_name = "images" if artifact_input_name == "images" else "x"
        result = tuned_model(**{input_name: images})
        output = next(iter(result.values())) if isinstance(result, dict) else result
        detections = output[0]
        if detections.ndim != 2 or detections.shape[1] != 6 or not torch.isfinite(detections).all():
            raise RuntimeError("Unexpected YOLO detection output")
        print(f"Validated Python inference: variant={variant}, output shape {tuple(output.shape)}", flush=True)
    finally:
        tuned_model.deactivate()


def main() -> None:
    """Parse arguments and run Python inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_tuned_model_path_arg(parser)
    add_variant_arg(parser)
    args = parser.parse_args()
    run_inference(args.tuned_model_path, args.variant)


if __name__ == "__main__":
    main()
