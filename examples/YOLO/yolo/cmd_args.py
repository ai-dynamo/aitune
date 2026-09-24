# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-line arguments shared by YOLO example commands."""

import argparse
from pathlib import Path


def add_tuned_model_path_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Use a working-directory checkpoint path throughout the example."""
    parser.add_argument(
        "--tuned-model-path",
        type=Path,
        default=Path("yolov10n.ait"),
        help="Path to the tuned AITune package (default: yolov10n.ait)",
    )
    return parser
