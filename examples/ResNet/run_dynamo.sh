#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Start the ResNet Dynamo HTTP frontend and backend worker, then run a
# classification request to verify the service.
#
# Usage:
#   ./run_dynamo.sh
#
export DYN_DISCOVERY_BACKEND=file
export DYN_EVENT_PLANE=zmq
export DYN_REQUEST_PLANE=tcp
export DYN_ROUTER_USE_KV_EVENTS=false

DISCOVERY_STORE=/tmp/dynamo_store_kv
rm -rf "$DISCOVERY_STORE"

echo "Starting the frontend..."
python -m resnet.dynamo.frontend &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

echo "Starting the backend..."
python -m resnet.dynamo.backend &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

trap 'kill "$FRONTEND_PID" "$BACKEND_PID" 2>/dev/null; rm -rf "$DISCOVERY_STORE"' EXIT

for i in {1..100}; do
  if curl -fsS http://localhost:8000/health | grep -q '"status":"ok"'; then
    break
  fi
  echo "Waiting for ResNet frontend health check... (attempt $i)"
  sleep 2
done

python -m resnet.dynamo.client --num-requests 1
