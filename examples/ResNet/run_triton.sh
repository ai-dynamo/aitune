#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

NVIDIA_RELEASE="${NVIDIA_RELEASE:-26.05}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRITON_IMAGE="nvcr.io/nvidia/tritonserver:$NVIDIA_RELEASE-py3"
MODEL_REPOSITORY="${MODEL_REPOSITORY:-$SCRIPT_DIR/model_repository}"
MODEL_NAME="${MODEL_NAME:-resnet50}"
if [[ ! -d "$MODEL_REPOSITORY/$MODEL_NAME" ]]; then
  echo "Missing $MODEL_REPOSITORY/$MODEL_NAME; run triton-model-store first" >&2
  exit 1
fi
MODEL_REPOSITORY_PATH="$(realpath "$MODEL_REPOSITORY")"

docker run --rm --gpus all --network host \
  -v "$MODEL_REPOSITORY_PATH:/models:ro" \
  "$TRITON_IMAGE" tritonserver --model-repository=/models &
TRITON_PID=$!
trap 'kill "$TRITON_PID" 2>/dev/null || true' EXIT

for attempt in {1..100}; do
  if curl -fsS http://localhost:8000/v2/health/ready >/dev/null 2>&1 && \
    curl -fsS "http://localhost:8000/v2/models/$MODEL_NAME/ready" >/dev/null 2>&1; then
    python -m resnet.triton.client --model-name "$MODEL_NAME" "$@"
    exit 0
  fi
  echo "Waiting for Triton... (attempt $attempt)"
  sleep 2
done

echo "Triton did not become ready" >&2
exit 1
