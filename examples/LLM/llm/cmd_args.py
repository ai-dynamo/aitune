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
"""Common command line arguments."""

import argparse


def get_tune_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Tune LLM model")
    parser.add_argument(
        "--model_id",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Model ID to load from Hugging Face Hub",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="static",
        choices=["dynamic", "static"],
        help="Cache implementation type",
    )
    return parser


def parse_sequence_lengths(value):
    """Parse sequence lengths from command line argument.

    Args:
        value: String in format "ISL,OSL ISL2,OSL2 ..." or "ISL,OSL" for single pair.

    Returns:
        list[tuple[int, int]]: List of (ISL, OSL) tuples.
    """
    pairs = value.split()
    result = []
    for pair in pairs:
        parts = pair.split(",")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"Each sequence length pair must be in format 'ISL,OSL', got '{pair}'")
        try:
            isl, osl = int(parts[0]), int(parts[1])
            result.append((isl, osl))
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"Sequence lengths must be integers, got '{pair}'") from e
    return result


def get_benchmark_parser():
    """Parse command line arguments for benchmarking.

    Returns:
        argparse.ArgumentParser: The parser with benchmark-specific arguments.
    """
    parser = argparse.ArgumentParser(description="Benchmark LLM model")
    parser.add_argument(
        "--model_id",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Model ID to load from Hugging Face Hub",
    )
    parser.add_argument(
        "--sequence_lengths",
        type=parse_sequence_lengths,
        default=[(128, 1024)],
        help="Sequence length pairs in format 'ISL,OSL' (e.g., '128,512' or '128,512 256,1024' for multiple pairs)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=1,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="static",
        choices=["no_cache", "dynamic", "static"],
        help="Cache implementation type",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="aot",
        choices=["vanilla", "aot"],
        help="Benchmark scenario to run",
    )
    return parser


def get_benchmark_all_parser():
    """Parse command line arguments for benchmarking all scenarios.

    Returns:
        argparse.ArgumentParser: The parser with benchmark all scenarios-specific arguments.
    """
    parser = argparse.ArgumentParser(description="Benchmark all scenarios of LLM model")
    parser.add_argument(
        "--model_id",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Model ID to load from Hugging Face Hub",
    )
    parser.add_argument(
        "--sequence_lengths",
        type=parse_sequence_lengths,
        default=[(1, 512), (1, 1024), (1, 2040)],  # 2040 avoid sliding window recompilation
        help="Sequence length pairs in format 'ISL,OSL' (e.g., '128,512' or '128,512 256,1024' for multiple pairs)",
    )
    parser.add_argument(
        "--run_baseline",
        action="store_true",
        help="Whether to run baseline scenario (no_cache, vanilla)",
    )

    return parser
