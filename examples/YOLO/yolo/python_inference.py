# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate a saved YOLO checkpoint through Python inference."""

import argparse
from pathlib import Path
from typing import cast

import torch

from aitune.torch import Module, load
from aitune.torch.module import OnnxModule
from yolo.cmd_args import add_tuned_model_path_arg
from yolo.model import sample_input


def run_inference(checkpoint: Path) -> None:
    """Check that the compiled checkpoint produces finite outputs."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing {checkpoint}; run yolo-tune first")

    images = sample_input()
    tuned_model = cast(Module, load(OnnxModule.for_checkpoint(), checkpoint))
    try:
        artifact_input_name = tuned_model.artifact().input_names[0]
        if artifact_input_name not in {"images", "input_x"}:
            raise RuntimeError(f"Unexpected YOLO input name: {artifact_input_name}")
        input_name = "images" if artifact_input_name == "images" else "x"
        result = tuned_model(**{input_name: images})
        output = next(iter(result.values())) if isinstance(result, dict) else result
        detections = output[0]
        if detections.ndim != 2 or detections.shape[1] != 6 or not torch.isfinite(detections).all():
            raise RuntimeError("Unexpected YOLO detection output")
        print(f"Validated Python inference: output shape {tuple(output.shape)}", flush=True)
    finally:
        tuned_model.deactivate()


def main() -> None:
    """Parse arguments and run Python inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_tuned_model_path_arg(parser)
    args = parser.parse_args()
    run_inference(args.tuned_model_path)


if __name__ == "__main__":
    main()
