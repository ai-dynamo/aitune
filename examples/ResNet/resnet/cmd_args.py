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
