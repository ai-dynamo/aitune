# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a Triton model store from a tuned ResNet package."""

import argparse
from pathlib import Path
from typing import cast

import aitune.triton
from aitune.torch import Module, load
from resnet.model import get_model


def get_parser():
    """Create the model-store generation parser."""
    parser = argparse.ArgumentParser(description="Generate a Triton model store from a tuned ResNet package")
    parser.add_argument("--model-name", default="resnet50", help="Name of the tuned ResNet model")
    parser.add_argument("--tuned-model-path", type=Path, default=Path("resnet50.ait"), help="Tuned AITune package")
    parser.add_argument(
        "--model-repository",
        type=Path,
        default=Path("model_repository"),
        help="Triton model repository (default: model_repository)",
    )
    return parser


def main():
    """Load a tuned package and generate its Triton deployment files."""
    args = get_parser().parse_args()
    tuned_model = cast(Module, load(get_model(args.model_name, pretrained=False), args.tuned_model_path))
    try:
        artifact = tuned_model.artifact()
        model_path = aitune.triton.publish(
            artifact,
            path=args.model_repository,
            model_name=args.model_name,
            max_batch_size=artifact.max_batch_size,
        )
        print(f"Triton model: {model_path}", flush=True)
        print(f"Model Analyzer configuration: {model_path / 'model_analyzer' / 'config.yaml'}", flush=True)
    finally:
        tuned_model.deactivate()


if __name__ == "__main__":
    main()
