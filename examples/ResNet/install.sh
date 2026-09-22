#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ -n "${TRT_VERSION:-}" ]]; then
  python3 -m pip install --extra-index-url https://pypi.nvidia.com "tensorrt==$TRT_VERSION"
else
  echo "Triton container package versions are unavailable; skipping container-specific installation"
fi

# The PyPI ONNX Runtime GPU wheel targets CUDA 12. Install the CUDA 13 build used by the Triton container.
python3 -m pip install \
  onnxruntime-gpu \
  --force-reinstall \
  --upgrade \
  --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/
