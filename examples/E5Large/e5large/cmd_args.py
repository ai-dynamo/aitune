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
"""Common command line arguments for E5Large."""

import argparse


def get_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="E5Large embedding model")
    parser.add_argument(
        "--model-name",
        type=str,
        default="intfloat/e5-large-v2",
        help="Model name, use only models compatible with SentenceTransformer",
    )
    parser.add_argument(
        "--tuned-model-path",
        type=str,
        default="e5large_tuned.pt",
        help="Path to save/load the tuned model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="query: how much protein should a female eat",
        help="Text prompt for embedding",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=4,
        help="Maximum batch size",
    )
    return parser
