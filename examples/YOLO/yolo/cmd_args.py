# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-line arguments shared by YOLO example commands."""

import argparse
from pathlib import Path


def add_output_path_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Use one default checkpoint path throughout the example."""
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "yolov10n.ait",
        help="Path to this example's AITune checkpoint",
    )
    return parser
