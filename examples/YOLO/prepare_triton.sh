#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
if [[ $# -ne 0 ]]; then
  echo "Usage: ./prepare_triton.sh" >&2
  exit 2
fi
NVIDIA_RELEASE="${NVIDIA_RELEASE:-26.05}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(realpath "$SCRIPT_DIR/../..")"

docker run --rm --gpus all --ipc host \
  --user "$(id -u):$(id -g)" \
  -e AITUNE_CACHE_DIR=/tmp/aitune-cache \
  -e HF_HOME=/tmp/huggingface-cache \
  -e TORCH_HOME=/tmp/torch-cache \
  -v "$REPOSITORY_ROOT:/workspace" \
  -w /workspace/examples/YOLO \
  "nvcr.io/nvidia/tritonserver:$NVIDIA_RELEASE-py3" \
  bash -lc '
    set -euo pipefail
    python3 -m venv --system-site-packages /tmp/aitune-yolo
    source /tmp/aitune-yolo/bin/activate
    pip install --extra-index-url https://pypi.nvidia.com \
      -e "/workspace[triton,torch212]" -e ".[triton]"
    ./install.sh
    yolo-tune
    yolo-triton-model-store
    ./run_triton.sh
  '
