# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Print versions of Python, torch, torch.cuda (driver/runtime), tensorrt, torch_tensorrt."""

# ruff: noqa: T201

import sys


def main():
    """Print versions of Python, torch, torch.cuda (driver/runtime), tensorrt, torch_tensorrt."""
    print("Python version:", sys.version)
    try:
        import torch

        print("torch version:", torch.__version__)
        print("torch.version.cuda:", torch.version.cuda)
        print("torch.cuda.is_available():", torch.cuda.is_available())
        # Try initializing CUDA context to ensure it works as in the pipeline
        try:
            torch.cuda.init()
            print("torch.cuda.init() succeeded.")
        except Exception as e:
            print("torch.cuda.init() failed:", e)
    except ImportError:
        print("torch not installed")
    try:
        import tensorrt

        print("tensorrt version:", tensorrt.__version__)
    except ImportError:
        print("tensorrt not installed")
    try:
        import torch_tensorrt

        print("torch_tensorrt version:", torch_tensorrt.__version__)
    except ImportError:
        print("torch_tensorrt not installed")


if __name__ == "__main__":
    main()
