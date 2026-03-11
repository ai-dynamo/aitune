# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common command line arguments for ResNet."""

import argparse
from pathlib import Path

default_image_path = str(Path(__file__).parent.parent / "dog.webp")


def get_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Tune ResNet model")
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet50",
        help="Name of the model to tune",
    )
    parser.add_argument(
        "--tuned-model-path",
        type=str,
        default="resnet50.ait",
        help="Path to save the tuned model",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=default_image_path,
        help="Path to the input image file",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=4,
        help="Maximum batch size (default: 4)",
    )
    return parser
