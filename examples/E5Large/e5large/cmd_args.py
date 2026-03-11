# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
