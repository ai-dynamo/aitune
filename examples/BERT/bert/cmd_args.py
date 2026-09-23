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


def add_output_path_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Use one default checkpoint path throughout the example."""
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "bert.ait",
        help="Path to this example's AITune checkpoint",
    )
    return parser
