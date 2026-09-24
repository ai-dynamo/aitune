# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-line arguments shared by BERT example commands."""

import argparse
from pathlib import Path

DEFAULT_MODEL_NAME = "bert-base-uncased"


def add_model_name_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Select the pretrained BERT model used for inputs and ONNX export."""
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Hugging Face BERT model identifier")
    return parser


def add_tuned_model_path_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Use a working-directory checkpoint path throughout the example."""
    parser.add_argument(
        "--tuned-model-path",
        type=Path,
        default=Path("bert.ait"),
        help="Path to the tuned AITune package (default: bert.ait)",
    )
    return parser
