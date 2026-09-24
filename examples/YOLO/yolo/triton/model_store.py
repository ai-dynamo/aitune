# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish a tuned YOLO checkpoint to a Triton model repository."""

import argparse
from pathlib import Path
from typing import cast

import aitune.triton
from aitune.torch import Module, load
from aitune.torch.module import onnx_checkpoint_placeholder
from yolo.cmd_args import add_tuned_model_path_arg


def create_model_store(checkpoint: Path, model_repository: Path) -> None:
    """Load the checkpoint and generate the Triton model and Analyzer config."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing {checkpoint}; run yolo-tune first")

    # These AOT backends restore their compiled artifacts without the original source model.
    tuned_model = cast(Module, load(onnx_checkpoint_placeholder(), checkpoint))
    try:
        model_path = aitune.triton.publish(tuned_model.artifact(), path=model_repository, model_name="yolov10n")
        print(f"Triton model: {model_path}", flush=True)
        print(f"Model Analyzer configuration: {model_path / 'model_analyzer' / 'config.yaml'}", flush=True)
    finally:
        tuned_model.deactivate()


def main() -> None:
    """Parse arguments and generate the Triton model store."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_tuned_model_path_arg(parser)
    parser.add_argument("--model-repository", type=Path, help="Triton model repository")
    args = parser.parse_args()
    create_model_store(
        args.tuned_model_path,
        args.model_repository or args.tuned_model_path.parent / "model_repository",
    )


if __name__ == "__main__":
    main()
