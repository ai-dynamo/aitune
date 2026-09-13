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

if [[ -z "${TRITON_NETWORK:-}" ]]; then
  TRITON_NETWORK=host
  if [[ -f /.dockerenv && -n "${CI_JOB_ID:-}" ]]; then
    # The Docker host's localhost is separate from the GitLab job's localhost.
    JOB_CONTAINER_ID="$(docker ps -q \
      --filter "label=com.gitlab.gitlab-runner.job.id=$CI_JOB_ID" \
      --filter "label=com.gitlab.gitlab-runner.type=build")"
    if [[ -z "$JOB_CONTAINER_ID" || "$JOB_CONTAINER_ID" == *$'\n'* ]]; then
      echo "Cannot identify the GitLab job container; set TRITON_NETWORK=container:<name-or-id>" >&2
      exit 1
    fi
    TRITON_NETWORK="container:$JOB_CONTAINER_ID"
  fi
fi

docker run --rm --gpus all --network "$TRITON_NETWORK" \
  -v "$MODEL_REPOSITORY_PATH:/models:ro" \
  "$TRITON_IMAGE" tritonserver --model-repository=/models --model-control-mode=none &
TRITON_PID=$!
trap 'kill "$TRITON_PID" 2>/dev/null || true' EXIT

for attempt in {1..100}; do
  if curl --noproxy '*' --connect-timeout 1 --max-time 2 -fsS \
    http://localhost:8000/v2/health/ready >/dev/null 2>&1 && \
    curl --noproxy '*' --connect-timeout 1 --max-time 2 -fsS \
      "http://localhost:8000/v2/models/$MODEL_NAME/ready" >/dev/null 2>&1; then
    python -m resnet.triton.client --model-name "$MODEL_NAME" "$@"
    exit 0
  fi
  echo "Waiting for Triton... (attempt $attempt)"
  sleep 2
done

echo "Triton did not become ready" >&2
curl --noproxy '*' --connect-timeout 1 --max-time 2 -fsS http://localhost:8000/v2/health/ready || true
curl --noproxy '*' --connect-timeout 1 --max-time 2 -fsS "http://localhost:8000/v2/models/$MODEL_NAME/ready" || true
exit 1
