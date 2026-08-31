#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Start the Dynamo HTTP frontend (OpenAI-compatible API, default port 8000), and the backend worker.
# Then, run the test script to verify the backend is running.
#
# Usage:
#   ./run_dynamo.sh
#
export DYN_DISCOVERY_BACKEND=file
export DYN_EVENT_PLANE=zmq
export DYN_REQUEST_PLANE=tcp
export DYN_ROUTER_USE_KV_EVENTS=false

echo "Starting the frontend..."
python -m dynamo.frontend --http-port 8000 & # (1) start the frontend in the background

FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

echo "Waiting for the frontend to start..."
sleep 2 # wait for the frontend to start

echo "Starting the backend..."
BACKEND_RANK_ZERO_PID_FILE=$(mktemp -t aitune-flux-dynamo-rank-zero.XXXXXX)
export AITUNE_DYNAMO_RANK_ZERO_PID_FILE="$BACKEND_RANK_ZERO_PID_FILE"
torchrun --standalone --nproc-per-node=gpu --module flux.dynamo.backend & # (2) start one backend rank per GPU

BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

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

echo "Waiting for the backend to start..."
sleep 2

for i in {1..100}; do
  if curl -s http://localhost:8000/health | grep -q '"dyn://aitune.backend.generate"'; then
    break
  fi
  echo "Waiting for dyn://aitune.backend.generate to appear in /health... (attempt $i)"
  sleep 10
done

for i in {1..5}; do
  if curl -s http://localhost:8000/v1/models | grep -q '"black-forest-labs/FLUX.1-dev"'; then
    break
  fi
  echo "Waiting for black-forest-labs/FLUX.1-dev to appear in /v1/models... (attempt $i)"
  sleep 2
done

python -m flux.dynamo.client # (3) run the test script with the shared defaults
