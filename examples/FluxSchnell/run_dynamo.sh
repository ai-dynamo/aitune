#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Start the Dynamo HTTP frontend (OpenAI-compatible API, default port 8000), and the backend worker.
# Run a smoke request, or keep the deployment and prompt UI running with --serve.
#
# Usage:
#   ./run_dynamo.sh [--serve]
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SERVE=false
case "${1:-}" in
  --serve) SERVE=true; shift ;;
  "") ;;
  *) echo "Usage: $0 [--serve]" >&2; exit 2 ;;
esac
if (($#)); then
  echo "Usage: $0 [--serve]" >&2
  exit 2
fi

export DYN_DISCOVERY_BACKEND=file
export DYN_EVENT_PLANE=zmq
export DYN_REQUEST_PLANE=tcp
export DYN_ROUTER_USE_KV_EVENTS=false

FRONTEND_PID=""
BACKEND_PID=""
WEB_PID=""
BACKEND_RANK_ZERO_PID_FILE=$(mktemp -t aitune-flux-schnell-dynamo-rank-zero.XXXXXX)
export AITUNE_DYNAMO_RANK_ZERO_PID_FILE="$BACKEND_RANK_ZERO_PID_FILE"
cleanup() {
  [[ -z "$WEB_PID" ]] || kill "$WEB_PID" 2>/dev/null || true
  if [[ -s "$BACKEND_RANK_ZERO_PID_FILE" ]]; then
    read -r BACKEND_RANK_ZERO_PID < "$BACKEND_RANK_ZERO_PID_FILE" || true
    kill "$BACKEND_RANK_ZERO_PID" 2>/dev/null || true
  elif [[ -n "$BACKEND_PID" ]]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  [[ -z "$BACKEND_PID" ]] || wait "$BACKEND_PID" 2>/dev/null || true
  [[ -z "$FRONTEND_PID" ]] || kill "$FRONTEND_PID" 2>/dev/null || true
  [[ -z "$FRONTEND_PID" ]] || wait "$FRONTEND_PID" 2>/dev/null || true
  [[ -z "$WEB_PID" ]] || wait "$WEB_PID" 2>/dev/null || true
  rm -f "$BACKEND_RANK_ZERO_PID_FILE"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Starting the frontend..."
python -m dynamo.frontend --http-port 8000 & # (1) start the frontend in the background

FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

echo "Waiting for the frontend to start..."
sleep 2 # wait for the frontend to start

echo "Starting the backend..."
torchrun --standalone --nproc-per-node=gpu --module flux_schnell.dynamo.backend & # (2) start one backend rank per GPU

BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

echo "Waiting for the backend to start..."
sleep 2

READY=false
for i in {1..100}; do
  if ! kill -0 "$FRONTEND_PID" 2>/dev/null || ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "Dynamo exited before becoming ready." >&2
    exit 1
  fi
  if curl -fsS --max-time 5 http://localhost:8000/health | grep -q '"dyn://aitune.backend.generate"'; then
    READY=true
    break
  fi
  echo "Waiting for dyn://aitune.backend.generate to appear in /health... (attempt $i)"
  sleep 10
done
if [[ "$READY" != true ]]; then
  echo "Timed out waiting for the Dynamo backend." >&2
  exit 1
fi

READY=false
for i in {1..5}; do
  if curl -fsS --max-time 5 http://localhost:8000/v1/models | grep -q '"black-forest-labs/FLUX.1-schnell"'; then
    READY=true
    break
  fi
  echo "Waiting for black-forest-labs/FLUX.1-schnell to appear in /v1/models... (attempt $i)"
  sleep 2
done
if [[ "$READY" != true ]]; then
  echo "FLUX.1-schnell was not registered in Dynamo." >&2
  exit 1
fi

if [[ "$SERVE" == true ]]; then
  python -m flux_schnell.dynamo.web --host "${WEB_HOST:-127.0.0.1}" --port "${WEB_PORT:-8080}" &
  WEB_PID=$!
  wait -n "$FRONTEND_PID" "$BACKEND_PID" "$WEB_PID"
else
  python -m flux_schnell.dynamo.client # (3) run the test script with the shared defaults
fi
