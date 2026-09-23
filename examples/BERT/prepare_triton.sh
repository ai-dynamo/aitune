#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
SOURCE="${1:?Usage: ./prepare_triton.sh torch|onnx}"
if [[ "$SOURCE" != torch && "$SOURCE" != onnx ]]; then
  echo "Source must be torch or onnx" >&2
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
  -e SOURCE="$SOURCE" \
  -e ONNX_PATH="${ONNX_PATH:-}" \
  -v "$REPOSITORY_ROOT:/workspace" \
  -w /workspace/examples/BERT \
  "nvcr.io/nvidia/tritonserver:$NVIDIA_RELEASE-py3" \
  bash -lc '
    python3 -m venv --system-site-packages /tmp/aitune-bert
    source /tmp/aitune-bert/bin/activate
    pip install --extra-index-url https://pypi.nvidia.com \
      -e "/workspace[triton,torch212]" -e ".[triton]"
    ./install.sh
    onnx_args=()
    if [[ -n "$ONNX_PATH" ]]; then onnx_args=(--onnx-path "$ONNX_PATH"); fi
    bert-tune --source "$SOURCE" "${onnx_args[@]}"
    bert-triton-model-store
    ./run_triton.sh
  '
