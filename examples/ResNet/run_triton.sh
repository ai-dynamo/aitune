#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Variant arguments also include tuning-only options. Only the image path is
# needed for this correctness check; the client supplies its default otherwise.
CLIENT_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image-path)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi
      CLIENT_ARGS+=("$1" "$2")
      shift 2
      ;;
    --image-path=*)
      CLIENT_ARGS+=("$1")
      shift
      ;;
    *)
      shift
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_REPOSITORY="${MODEL_REPOSITORY:-$SCRIPT_DIR/model_repository}"
MODEL_NAME="${MODEL_NAME:-resnet50}"
TRITONSERVER="${TRITONSERVER:-/opt/tritonserver/bin/tritonserver}"
PERF_ANALYZER="${PERF_ANALYZER:-$(command -v perf_analyzer || true)}"
if [[ ! -f "$MODEL_REPOSITORY/$MODEL_NAME/config.pbtxt" ]]; then
  echo "Missing $MODEL_REPOSITORY/$MODEL_NAME/config.pbtxt; run triton-model-store first" >&2
  exit 1
fi
MODEL_REPOSITORY_PATH="$(realpath "$MODEL_REPOSITORY")"
MODEL_ANALYZER_CONFIG="$MODEL_REPOSITORY_PATH/$MODEL_NAME/model_analyzer/config.yaml"
MODEL_ANALYZER_WORKSPACE="${MODEL_REPOSITORY_PATH}-model-analyzer/$MODEL_NAME"
DEPLOYMENT_MODEL_REPOSITORY="${DEPLOYMENT_MODEL_REPOSITORY:-$MODEL_REPOSITORY_PATH-deployment}"
TRITON_LOG_PATH="${TRITON_LOG_PATH:-$(dirname "$MODEL_REPOSITORY_PATH")/tritonserver.log}"
if [[ ! -x "$TRITONSERVER" ]]; then
  echo "Missing Triton server executable: $TRITONSERVER" >&2
  exit 1
fi
if [[ -z "$PERF_ANALYZER" || ! -x "$PERF_ANALYZER" ]]; then
  echo "Missing Perf Analyzer executable; install perf-analyzer or set PERF_ANALYZER" >&2
  exit 1
fi

# Model Analyzer creates its output directories but requires their shared parent to exist.
mkdir -p "$MODEL_ANALYZER_WORKSPACE"
model-analyzer profile \
  --config-file "$MODEL_ANALYZER_CONFIG" \
  --triton-server-path "$TRITONSERVER" \
  --perf-analyzer-path "$PERF_ANALYZER"
python3 -m resnet.triton.promote \
  --analyzer-config "$MODEL_ANALYZER_CONFIG" \
  --deployment-model-repository "$DEPLOYMENT_MODEL_REPOSITORY"
MODEL_REPOSITORY_PATH="$(realpath "$DEPLOYMENT_MODEL_REPOSITORY")"

"$TRITONSERVER" --model-repository="$MODEL_REPOSITORY_PATH" --model-control-mode=none >"$TRITON_LOG_PATH" 2>&1 &
TRITON_PID=$!
trap 'kill "$TRITON_PID" >/dev/null 2>&1 || true' EXIT

for attempt in {1..100}; do
  if curl --noproxy '*' --connect-timeout 1 --max-time 2 -fsS \
    http://localhost:8000/v2/health/ready >/dev/null 2>&1 && \
    curl --noproxy '*' --connect-timeout 1 --max-time 2 -fsS \
      "http://localhost:8000/v2/models/$MODEL_NAME/ready" >/dev/null 2>&1; then
    cat "$TRITON_LOG_PATH"
    python3 -m resnet.triton.client --model-name "$MODEL_NAME" "${CLIENT_ARGS[@]}"
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
