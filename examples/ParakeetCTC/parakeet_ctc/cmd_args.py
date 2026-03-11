# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common command line arguments for ParakeetCTC."""

import argparse
from pathlib import Path

from parakeet_ctc.sample_data import ensure_sample_audio


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        required=False,
        default="nvidia/parakeet-ctc-0.6b",
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
        default="parakeet_ctc_0.6b_tuned.ait",
        help="Saved model file path",
    )
    return parser.parse_args()
