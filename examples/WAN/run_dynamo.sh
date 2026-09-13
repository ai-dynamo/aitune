#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

export DYN_DISCOVERY_BACKEND=file
export DYN_EVENT_PLANE=zmq
export DYN_REQUEST_PLANE=tcp
export DYN_ROUTER_USE_KV_EVENTS=false

if ! EXPECTED_MODEL=$(python - <<'PY'
from wan.defaults import DEFAULT_MODEL_NAME
from wan.dynamo.config import load_config

print(load_config().get("model_name", DEFAULT_MODEL_NAME))
PY
); then
  echo "Failed to load the configured WAN model name" >&2
  exit 1
fi

echo "Starting the frontend..."
python -m dynamo.frontend --http-port 8000 &
FRONTEND_PID=$!

echo "Starting the backend..."
BACKEND_RANK_ZERO_PID_FILE=$(mktemp -t aitune-wan-dynamo-rank-zero.XXXXXX)
export AITUNE_DYNAMO_RANK_ZERO_PID_FILE="$BACKEND_RANK_ZERO_PID_FILE"
torchrun --standalone --nproc-per-node=gpu --module wan.dynamo.backend &
BACKEND_PID=$!

cleanup() {
  if [[ -s "$BACKEND_RANK_ZERO_PID_FILE" ]]; then
    read -r BACKEND_RANK_ZERO_PID < "$BACKEND_RANK_ZERO_PID_FILE"
    kill "$BACKEND_RANK_ZERO_PID" 2>/dev/null || true
  else
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  wait "$BACKEND_PID" 2>/dev/null || true

  kill "$FRONTEND_PID" 2>/dev/null || true
  wait "$FRONTEND_PID" 2>/dev/null || true
  rm -f "$BACKEND_RANK_ZERO_PID_FILE"
}
trap cleanup EXIT

echo "Waiting for the WAN endpoint..."
endpoint_ready=false
for i in {1..100}; do
  if curl --silent --max-time 5 http://localhost:8000/health | grep -q '"dyn://aitune.backend.generate"'; then
    endpoint_ready=true
    break
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    wait "$BACKEND_PID"
    backend_status=$?
    if [[ "$backend_status" -eq 0 ]]; then
      backend_status=1
    fi
    echo "WAN backend exited before registering its endpoint (status $backend_status)" >&2
    exit "$backend_status"
  fi
  echo "Waiting for dyn://aitune.backend.generate to appear in /health... (attempt $i)"
  sleep 10
done
if [[ "$endpoint_ready" != true ]]; then
  echo "Timed out waiting for dyn://aitune.backend.generate in /health" >&2
  exit 1
fi

model_ready=false
for i in {1..5}; do
  if curl --silent --max-time 5 http://localhost:8000/v1/models \
    | grep -Fq "\"$EXPECTED_MODEL\""; then
    model_ready=true
    break
  fi
  echo "Waiting for $EXPECTED_MODEL to appear in /v1/models... (attempt $i)"
  sleep 2
done
if [[ "$model_ready" != true ]]; then
  echo "Timed out waiting for $EXPECTED_MODEL in /v1/models" >&2
  exit 1
fi

python -m wan.dynamo.client
