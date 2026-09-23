#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="yolov10n"
MODEL_REPOSITORY="${MODEL_REPOSITORY:-$SCRIPT_DIR/artifacts/model_repository}"
TRITONSERVER="${TRITONSERVER:-/opt/tritonserver/bin/tritonserver}"
PERF_ANALYZER="${PERF_ANALYZER:-$(command -v perf_analyzer || true)}"
if [[ ! -f "$MODEL_REPOSITORY/$MODEL_NAME/config.pbtxt" ]]; then
  echo "Missing model config; run yolo-triton-model-store first" >&2
  exit 1
fi
if [[ ! -x "$TRITONSERVER" || -z "$PERF_ANALYZER" || ! -x "$PERF_ANALYZER" ]]; then
  echo "Triton server and Perf Analyzer executables are required" >&2
  exit 1
fi

MODEL_REPOSITORY="$(realpath "$MODEL_REPOSITORY")"
ANALYZER_CONFIG="$MODEL_REPOSITORY/$MODEL_NAME/model_analyzer/config.yaml"
ANALYZER_WORKSPACE="$MODEL_REPOSITORY-model-analyzer/$MODEL_NAME"
DEPLOYMENT_REPOSITORY="${DEPLOYMENT_REPOSITORY:-$MODEL_REPOSITORY-deployment}"
TRITON_LOG="${TRITON_LOG:-$(dirname "$MODEL_REPOSITORY")/tritonserver.log}"
mkdir -p "$ANALYZER_WORKSPACE"
python3 -m yolo.python_inference
model-analyzer profile --config-file "$ANALYZER_CONFIG" \
  --triton-server-path "$TRITONSERVER" --perf-analyzer-path "$PERF_ANALYZER"
python3 -m yolo.triton.promote \
  --analyzer-config "$ANALYZER_CONFIG" --deployment-model-repository "$DEPLOYMENT_REPOSITORY"

"$TRITONSERVER" --model-repository="$(realpath "$DEPLOYMENT_REPOSITORY")" \
  --model-control-mode=none >"$TRITON_LOG" 2>&1 &
TRITON_PID=$!
trap 'kill "$TRITON_PID" >/dev/null 2>&1 || true' EXIT
for attempt in {1..100}; do
  if curl --noproxy '*' --connect-timeout 1 --max-time 2 -fsS \
    http://localhost:8000/v2/health/ready >/dev/null 2>&1 && \
    curl --noproxy '*' --connect-timeout 1 --max-time 2 -fsS \
      "http://localhost:8000/v2/models/$MODEL_NAME/ready" >/dev/null 2>&1; then
    python3 -m yolo.triton.client
    exit 0
  fi
  if ! kill -0 "$TRITON_PID" >/dev/null 2>&1; then
    echo "Triton exited before $MODEL_NAME became ready" >&2
    break
  fi
  sleep 2
done
cat "$TRITON_LOG" >&2
echo "Triton model $MODEL_NAME did not become ready" >&2
exit 1
