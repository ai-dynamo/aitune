#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ -n "${TRT_VERSION:-}" ]]; then
  pip install --extra-index-url https://pypi.nvidia.com "tensorrt==$TRT_VERSION"
else
  echo "TRT_VERSION is unset; skipping container-specific TensorRT installation"
fi

# The PyPI ONNX Runtime GPU wheel targets CUDA 12, while current Triton images use CUDA 13.
onnxruntime_wheel_dir="$(mktemp -d)"
trap 'rm -r -- "$onnxruntime_wheel_dir"' EXIT
pip download --no-deps --dest "$onnxruntime_wheel_dir" \
  --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ \
  onnxruntime-gpu
pip install --force-reinstall "$onnxruntime_wheel_dir"/onnxruntime_gpu-*.whl 'protobuf>=6.33.5,<7'
