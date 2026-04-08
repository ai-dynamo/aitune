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

echo "Starting the frontend..."
python -m dynamo.frontend --http-port 8000 & # (1) start the frontend in the background

FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

echo "Waiting for the frontend to start..."
sleep 2 # wait for the frontend to start

echo "Starting the backend..."
python -m flux.dynamo.backend & # (2) start the backend in the background

BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

trap "kill -9 $FRONTEND_PID; kill -9 $BACKEND_PID" EXIT

echo "Waiting for the backend to start..."
sleep 2

for i in {1..10}; do
  if curl -s http://localhost:8000/health | grep -q '"dyn://aitune.backend.generate"'; then
    break
  fi
  echo "Waiting for dyn://aitune.backend.generate to appear in /health... (attempt $i)"
  sleep 5
done

for i in {1..5}; do
  if curl -s http://localhost:8000/v1/models | grep -q '"black-forest-labs/FLUX.1-dev"'; then
    break
  fi
  echo "Waiting for black-forest-labs/FLUX.1-dev to appear in /v1/models... (attempt $i)"
  sleep 2
done

python -m flux.dynamo.client --prompt "A futuristic cityscape at night" # (3) run the test script
