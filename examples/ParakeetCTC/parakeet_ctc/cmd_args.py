# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
"""Common command line arguments for ResNet."""

import argparse
from pathlib import Path


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
        required=True,
        help="Audio path",
    )
    parser.add_argument(
        "--tuned-model-path",
        type=Path,
        required=False,
        default="parakeet_ctc_0.6b_tuned.ait",
        help="Saved model file path",
    )
    return parser.parse_args()
