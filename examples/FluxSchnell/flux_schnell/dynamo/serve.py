# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch Dynamo and optionally keep the recipe's prompt UI running."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from ..config import argument_parser, load_recipe


def _ready(recipe):
    try:
        with urlopen(recipe.endpoint + "/health", timeout=5) as response:
            health = response.read().decode()
        with urlopen(recipe.endpoint + "/v1/models", timeout=5) as response:
            models = json.load(response)["data"]
        return '"dyn://aitune.backend.generate"' in health and any(
            model["id"] == recipe["model"]["base_model"] for model in models
        )
    except (URLError, OSError, ValueError, KeyError):
        return False


def _wait_ready(recipe, processes):
    deadline = time.monotonic() + recipe["deployment"]["readiness_timeout_s"]
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("A Dynamo process exited before readiness")
        if _ready(recipe):
            return
        print("Waiting for the recipe's Dynamo model to become ready...", flush=True)
        time.sleep(10)
    raise TimeoutError("Dynamo readiness exceeded deployment.readiness_timeout_s")


def _stop(process):
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run(recipe, serve):
    env = {
        **os.environ,
        "DYN_DISCOVERY_BACKEND": "file",
        "DYN_EVENT_PLANE": "zmq",
        "DYN_REQUEST_PLANE": "tcp",
        "DYN_ROUTER_USE_KV_EVENTS": "false",
    }
    processes = []
    backend = None
    with tempfile.TemporaryDirectory(prefix="flux-schnell-dynamo-") as directory:
        pid_file = Path(directory) / "rank-zero.pid"
        env["AITUNE_DYNAMO_RANK_ZERO_PID_FILE"] = str(pid_file)

        def start(module, *arguments):
            process = subprocess.Popen([sys.executable, "-m", module, *arguments], env=env, start_new_session=True)
            processes.append(process)
            return process

        try:
            start("dynamo.frontend", "--http-port", str(recipe["deployment"]["port"]))
            backend = start("flux_schnell.dynamo.backend", "--recipe", str(recipe.path))
            _wait_ready(recipe, processes)
            if serve:
                start("flux_schnell.dynamo.web", "--recipe", str(recipe.path))
                while all(process.poll() is None for process in processes):
                    time.sleep(1)
                raise RuntimeError("A deployment process exited")
            subprocess.run(
                [sys.executable, "-m", "flux_schnell.dynamo.client", "--recipe", str(recipe.path)], check=True
            )
        finally:
            if backend is not None and pid_file.is_file() and backend.poll() is None:
                try:
                    os.kill(int(pid_file.read_text()), signal.SIGTERM)
                    backend.wait(timeout=60)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
            for process in reversed(processes):
                _stop(process)


def main():
    parser = argument_parser(__doc__)
    parser.add_argument(
        "--serve", action="store_true", help="Keep Dynamo and the UI running instead of running the client"
    )
    args = parser.parse_args()
    recipe = load_recipe(args.recipe)

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        run(recipe, args.serve)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
