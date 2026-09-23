# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish a tuned YOLO checkpoint to a Triton model repository."""

import argparse
from pathlib import Path
from typing import cast

import torch

import aitune.triton
from aitune.torch import Module, load


def main() -> None:
    """Load the checkpoint and generate the Triton model and Analyzer config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Directory used by yolo-tune")
    parser.add_argument("--model-repository", type=Path, help="Triton model repository")
    args = parser.parse_args()

    output_dir = args.output_dir or Path(__file__).resolve().parents[2] / "artifacts"
    model_repository = args.model_repository or output_dir / "model_repository"
    checkpoint = output_dir / "yolov10n.ait"
    if not checkpoint.is_file():
        parser.error(f"Missing {checkpoint}; run yolo-tune first")

    # These AOT backends restore their compiled artifacts without the original source model.
    tuned_model = cast(Module, load(torch.nn.Identity(), checkpoint))
    try:
        model_path = aitune.triton.publish(
            tuned_model.artifact(),
            path=model_repository,
            model_name="yolov10n",
            dynamic_batching=False,
        )
        print(f"Triton model: {model_path}", flush=True)
        print(f"Model Analyzer configuration: {model_path / 'model_analyzer' / 'config.yaml'}", flush=True)
    finally:
        tuned_model.deactivate()


if __name__ == "__main__":
    main()
