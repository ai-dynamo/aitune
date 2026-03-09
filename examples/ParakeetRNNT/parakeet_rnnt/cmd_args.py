# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
