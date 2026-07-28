#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Start the ParakeetRNNT Dynamo HTTP frontend (default port 8000), and the backend worker.
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
python -m parakeet_rnnt.dynamo.frontend & # (1) start the frontend in the background

FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

echo "Waiting for the frontend to start..."
sleep 2 # wait for the frontend to start

echo "Starting the backend..."
python -m parakeet_rnnt.dynamo.backend & # (2) start the backend in the background

BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

trap "kill -9 $FRONTEND_PID; kill -9 $BACKEND_PID" EXIT

echo "Waiting for the backend to start..."
sleep 2

for i in {1..100}; do
  if curl -s http://localhost:8000/health | grep -q '"status":"ok"'; then
    break
  fi
  echo "Waiting for ParakeetRNNT frontend health check... (attempt $i)"
  sleep 5
done

python -m parakeet_rnnt.dynamo.client --num-requests 1 # (3) run the test script
