# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Print the GPU software stack versions and fail if CUDA or a required package does not work."""

# ruff: noqa: T201

import sys


def main():
    """Print the GPU software stack versions and fail if CUDA or a required package does not work."""
    print("Python version:", sys.version)
    import torch

    print("torch version:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    torch.cuda.init()
    print("torch.cuda.device_count():", torch.cuda.device_count())

    import tensorrt

    print("tensorrt version:", tensorrt.__version__)

    import torch_tensorrt

    print("torch_tensorrt version:", torch_tensorrt.__version__)


if __name__ == "__main__":
    main()
