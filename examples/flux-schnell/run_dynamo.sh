#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

config=recipes/single-gpu.yaml
variant=aitune
source_args=()
while (($#)); do
    case "$1" in
        --config) config=$2; shift 2 ;;
        --variant) variant=$2; shift 2 ;;
        --model-id|--checkpoint) source_args+=("$1" "$2"); shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--config RECIPE] [--variant aitune|original] [--model-id ID | --checkpoint DIR]"
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [[ "$variant" != aitune && "$variant" != original ]]; then
    echo "--variant must be aitune or original" >&2
    exit 2
fi

read -r gpu_count port timeout name < <(uv run --frozen python -c '
import socket, sys
from flux_schnell_recipe.config import load_config
cfg = load_config(sys.argv[1])
with socket.socket() as probe:
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    probe.bind(("127.0.0.1", cfg["deployment"]["port"]))
print(cfg["execution"]["gpu_count"], cfg["deployment"]["port"], cfg["deployment"]["readiness_timeout_s"], cfg["name"])
' "$config")

export DYN_DISCOVERY_BACKEND=file DYN_EVENT_PLANE=zmq DYN_REQUEST_PLANE=tcp
export DYN_ROUTER_USE_KV_EVENTS=false DYN_SYSTEM_PORT=-1 DYN_TCP_RPC_HOST=127.0.0.1
export PYTHONUNBUFFERED=1
# Dynamo 1.5 expects an interface name here; lo keeps response streams on loopback.
export DYN_TCP_RESPONSE_STREAM_HOST=lo
export FLUX_RUN_ID="$$_$(date +%s)"
export FLUX_RANK0_PID_FILE
FLUX_RANK0_PID_FILE=$(mktemp)
frontend_pid= backend_pid=

cleanup() {
    trap - EXIT INT TERM
    # Rank zero broadcasts Dynamo shutdown before torchrun exits.
    if [[ -s "$FLUX_RANK0_PID_FILE" ]]; then
        kill -TERM "$(cat "$FLUX_RANK0_PID_FILE")" 2>/dev/null || echo "Rank zero already stopped"
        for ((i=0; i<20; i++)); do
            [[ -n "$backend_pid" ]] && kill -0 "$backend_pid" 2>/dev/null || break
            sleep 1
        done
    fi
    for pid in "$backend_pid" "$frontend_pid"; do
        [[ -n "$pid" ]] || continue
        if kill -0 -- "-$pid" 2>/dev/null; then
            kill -TERM -- "-$pid" 2>/dev/null || echo "Process group $pid already stopped"
            sleep 1
            kill -KILL -- "-$pid" 2>/dev/null || echo "Process group $pid stopped"
        fi
        wait "$pid" 2>/dev/null || echo "Process $pid exited during shutdown"
    done
    rm -f -- "$FLUX_RANK0_PID_FILE"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

setsid uv run --frozen python -m dynamo.frontend --http-host 127.0.0.1 --http-port "$port" &
frontend_pid=$!
launcher=(python -m)
if ((gpu_count > 1)); then
    launcher=(torchrun --standalone --nproc-per-node="$gpu_count" --max-restarts=0 --module)
fi
setsid uv run --frozen "${launcher[@]}" flux_schnell_recipe.dynamo.backend \
    --config "$config" --variant "$variant" "${source_args[@]}" &
backend_pid=$!

model="flux-schnell-$variant-$name-$FLUX_RUN_ID"
deadline=$((SECONDS + timeout))
echo "Waiting for model loading, compilation and warmup (timeout: ${timeout}s)..."
until curl -fsS "http://127.0.0.1:$port/v1/models" 2>/dev/null | uv run --frozen python -c \
    'import json, sys; sys.exit(sys.argv[1] not in {m["id"] for m in json.load(sys.stdin)["data"]})' "$model" 2>/dev/null; do
    if ! kill -0 "$frontend_pid" "$backend_pid" 2>/dev/null; then
        echo "A Dynamo process exited; see its output above" >&2
        exit 1
    fi
    if ((SECONDS >= deadline)); then
        echo "Dynamo readiness timed out" >&2
        exit 1
    fi
    sleep 2
done
echo "Ready at http://127.0.0.1:$port; model: $model; stop with Ctrl-C"
wait -n "$frontend_pid" "$backend_pid"
