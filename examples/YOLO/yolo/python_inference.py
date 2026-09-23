# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate a saved YOLO checkpoint through Python inference."""

import argparse
from pathlib import Path
from typing import cast

import torch

from aitune.torch import Module, load
from yolo.tune import _image, _processor


@torch.inference_mode()
def main() -> None:
    """Check that the compiled checkpoint produces detections."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Directory used by yolo-tune")
    args = parser.parse_args()

    output_dir = args.output_dir or Path(__file__).resolve().parents[1] / "artifacts"
    checkpoint = output_dir / "yolov10n.ait"
    if not checkpoint.is_file():
        parser.error(f"Missing {checkpoint}; run yolo-tune first")

    images = _image(_processor())
    tuned_model = cast(Module, load(torch.nn.Identity(), checkpoint))
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
        detections = detections[detections[:, 4] >= 0.4]
        if not len(detections):
            raise RuntimeError("The street image produced no detections above the 0.4 confidence threshold")
        print(f"Validated Python inference: {len(detections)} detections above confidence 0.4", flush=True)
    finally:
        tuned_model.deactivate()


if __name__ == "__main__":
    main()
