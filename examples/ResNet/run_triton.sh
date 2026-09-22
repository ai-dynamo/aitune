#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_REPOSITORY="${MODEL_REPOSITORY:-$SCRIPT_DIR/model_repository}"
MODEL_NAME="${MODEL_NAME:-resnet50}"
TRITONSERVER="${TRITONSERVER:-/opt/tritonserver/bin/tritonserver}"
if [[ ! -f "$MODEL_REPOSITORY/$MODEL_NAME/config.pbtxt" ]]; then
  echo "Missing $MODEL_REPOSITORY/$MODEL_NAME/config.pbtxt; run triton-model-store first" >&2
  exit 1
fi
MODEL_REPOSITORY_PATH="$(realpath "$MODEL_REPOSITORY")"
TRITON_LOG_PATH="${TRITON_LOG_PATH:-$(dirname "$MODEL_REPOSITORY_PATH")/tritonserver.log}"
if [[ ! -x "$TRITONSERVER" ]]; then
  echo "Missing Triton server executable: $TRITONSERVER" >&2
  exit 1
fi

"$TRITONSERVER" --model-repository="$MODEL_REPOSITORY_PATH" --model-control-mode=none >"$TRITON_LOG_PATH" 2>&1 &
TRITON_PID=$!
trap 'kill "$TRITON_PID" >/dev/null 2>&1 || true' EXIT

for attempt in {1..100}; do
  if curl --noproxy '*' --connect-timeout 1 --max-time 2 -fsS \
    http://localhost:8000/v2/health/ready >/dev/null 2>&1 && \
    curl --noproxy '*' --connect-timeout 1 --max-time 2 -fsS \
      "http://localhost:8000/v2/models/$MODEL_NAME/ready" >/dev/null 2>&1; then
    cat "$TRITON_LOG_PATH"
    python -m resnet.triton.client --model-name "$MODEL_NAME" "$@"
    exit 0
  fi
  if ! kill -0 "$TRITON_PID" >/dev/null 2>&1; then
    echo "Triton exited before the model became ready" >&2
    break
  fi
  echo "Waiting for Triton... (attempt $attempt)"
  sleep 2
done

echo "Triton did not become ready" >&2
cat "$TRITON_LOG_PATH" >&2
curl --noproxy '*' --connect-timeout 1 --max-time 2 -fsS http://localhost:8000/v2/health/ready || true
curl --noproxy '*' --connect-timeout 1 --max-time 2 -fsS "http://localhost:8000/v2/models/$MODEL_NAME/ready" || true
exit 1
