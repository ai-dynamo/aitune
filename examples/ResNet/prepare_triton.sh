#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

NVIDIA_RELEASE="${NVIDIA_RELEASE:-26.05}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(realpath "$SCRIPT_DIR/../..")"

docker run --rm --gpus all --ipc host \
  --user "$(id -u):$(id -g)" \
  -e AITUNE_CACHE_DIR=/tmp/aitune-cache \
  -e HF_HOME=/tmp/huggingface-cache \
  -e TORCH_HOME=/tmp/torch-cache \
  -v "$REPOSITORY_ROOT:/workspace" \
  -w /workspace/examples/ResNet \
  "nvcr.io/nvidia/tritonserver:$NVIDIA_RELEASE-py3" \
  bash -lc '
    python -m venv --system-site-packages /tmp/aitune-resnet
    source /tmp/aitune-resnet/bin/activate
    python -m pip install --extra-index-url https://pypi.nvidia.com \
      -e "/workspace[triton,torch212]" \
      -e /workspace/examples/common \
      -e ".[triton]"
    tune --target triton
    triton-model-store
    ./run_triton.sh
  '
