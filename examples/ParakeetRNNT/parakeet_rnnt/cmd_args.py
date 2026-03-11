# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common command line arguments for ParakeetRNNT."""

import argparse
from pathlib import Path

from parakeet_rnnt.sample_data import ensure_sample_audio


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        required=False,
        default="nvidia/parakeet-rnnt-1.1b",
        help="Model name",
    )
    parser.add_argument(
        "-a",
        "--audio_path",
        type=Path,
        default=ensure_sample_audio(),
        help="Audio path (downloaded automatically if not provided)",
    )
    parser.add_argument(
        "--tuned-model-path",
        type=Path,
        required=False,
        default="parakeet_rnnt_1.1b_tuned.pt",
        help="Saved model file path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=64,
        help="Batch size",
    )
    return parser.parse_args()
