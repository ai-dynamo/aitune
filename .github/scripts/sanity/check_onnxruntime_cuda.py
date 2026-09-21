# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify that onnxruntime-gpu is installed and can be used."""

# ruff: noqa: T201

import argparse
import io
import re
import sys

import onnxruntime as ort


def main(cuda_major_version: int) -> None:
    """Verify that onnxruntime-gpu is installed and can be used.

    Args:
        cuda_major_version: The major version of the CUDA runtime to use.
    """
    print("onnxruntime-gpu version:", ort.__version__)

    ort_debug_info = capture_print(ort.print_debug_info)
    match = re.search(r"CUDA version used in build:(.*)$", ort_debug_info, re.MULTILINE)
    if match:
        cuda_version = match.group(1).strip()
        print("onnxruntime-gpu CUDA build version:", cuda_version)
        if not cuda_version.startswith(f"{cuda_major_version}"):
            print(f"CUDA major version mismatch: {cuda_version} != {cuda_major_version}.x")
            sys.exit(1)
    else:
        print("Could not find CUDA version in onnxruntime debug info.")
        sys.exit(1)


def capture_print(func, *args, **kwargs):
    """Capture stdout from a function and return the output as a string."""
    buf = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = buf
        func(*args, **kwargs)
        return buf.getvalue()
    finally:
        sys.stdout = old_stdout


if __name__ == "__main__":
    args = argparse.ArgumentParser(usage="check_onnxruntime_cuda.py --cuda-major-version <cuda-major-version>")
    args.add_argument("--cuda-major-version", type=int, required=True)
    args = args.parse_args()
    main(args.cuda_major_version)
