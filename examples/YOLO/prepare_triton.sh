#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
VARIANT=fp16
if [[ $# -eq 2 && "$1" == --variant ]]; then
  VARIANT="$2"
elif [[ $# -eq 1 && "$1" == --variant=* ]]; then
  VARIANT="${1#--variant=}"
elif [[ $# -ne 0 ]]; then
  echo "Usage: ./prepare_triton.sh [--variant fp16|int8]" >&2
  exit 2
fi
case "$VARIANT" in
  fp16|int8) ;;
  *) echo "Unknown YOLO variant: $VARIANT; choose fp16 or int8" >&2; exit 2 ;;
esac
NVIDIA_RELEASE="${NVIDIA_RELEASE:-26.05}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(realpath "$SCRIPT_DIR/../..")"

docker run --rm --gpus all --ipc host \
  --user "$(id -u):$(id -g)" \
  -e YOLO_VARIANT="$VARIANT" \
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
    yolo-tune --variant "$YOLO_VARIANT"
    yolo-triton-model-store
    ./run_triton.sh --variant "$YOLO_VARIANT"
  '
