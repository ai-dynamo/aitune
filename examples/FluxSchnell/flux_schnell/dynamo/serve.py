# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve Dynamo and the prompt UI on one GPU, or run a smoke test and shut down."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from logging import basicConfig, getLogger
from pathlib import Path
from types import FrameType
from typing import NoReturn
from urllib.error import URLError
from urllib.request import urlopen

from ..config import Recipe, argument_parser, load_recipe

logger = getLogger(__name__)


def _ready(recipe: Recipe) -> bool:
    try:
        with urlopen(recipe.endpoint + "/health", timeout=5) as response:
            health = response.read().decode()
        with urlopen(recipe.endpoint + "/v1/models", timeout=5) as response:
            models = json.load(response)["data"]
        return '"dyn://aitune.backend.generate"' in health and any(
            model["id"] == recipe.model.base_model for model in models
        )
    except (URLError, OSError, ValueError, KeyError) as exc:
        logger.debug("Dynamo readiness check failed at %s: %s", recipe.endpoint, exc)
        return False


def _wait_ready(recipe: Recipe, processes: Sequence[subprocess.Popen[bytes]]) -> None:
    deadline = time.monotonic() + recipe.deployment.readiness_timeout_s
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("A Dynamo process exited before readiness")
        if _ready(recipe):
            return
        print("Waiting for the recipe's Dynamo model to become ready...", flush=True)
        time.sleep(10)
    raise TimeoutError("Dynamo readiness exceeded deployment.readiness_timeout_s")


def _stop(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        logger.debug("Process group %s already exited before SIGTERM", process.pid)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        logger.warning("Process %s did not stop within 60 seconds; sending SIGKILL to its group", process.pid)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            logger.debug("Process group %s exited before SIGKILL", process.pid)
        process.wait()


def _serve(recipe: Recipe, smoke_test: bool) -> None:
    env = {
        **os.environ,
        "DYN_DISCOVERY_BACKEND": "file",
        "DYN_EVENT_PLANE": "zmq",
        "DYN_REQUEST_PLANE": "tcp",
        "DYN_ROUTER_USE_KV_EVENTS": "false",
    }
    processes: list[subprocess.Popen[bytes]] = []
    backend: subprocess.Popen[bytes] | None = None
    with tempfile.TemporaryDirectory(prefix="flux-schnell-dynamo-") as directory:
        pid_file = Path(directory) / "rank-zero.pid"
        env["AITUNE_DYNAMO_RANK_ZERO_PID_FILE"] = str(pid_file)

        def start(module: str, *arguments: str) -> subprocess.Popen[bytes]:
            process = subprocess.Popen([sys.executable, "-m", module, *arguments], env=env, start_new_session=True)
            processes.append(process)
            return process

        try:
            start("dynamo.frontend", "--http-port", str(recipe.deployment.port))
            backend = start("flux_schnell.dynamo.backend", "--recipe", str(recipe.path))
            _wait_ready(recipe, processes)
            if smoke_test:
                subprocess.run(
                    [sys.executable, "-m", "flux_schnell.dynamo.client", "--recipe", str(recipe.path)], check=True
                )
            else:
                start("flux_schnell.dynamo.web", "--recipe", str(recipe.path))
                while all(process.poll() is None for process in processes):
                    time.sleep(1)
                raise RuntimeError("A deployment process exited")
        finally:
            if backend is not None and pid_file.is_file() and backend.poll() is None:
                try:
                    os.kill(int(pid_file.read_text()), signal.SIGTERM)
                    backend.wait(timeout=60)
                except ProcessLookupError:
                    logger.debug("Rank-zero process already exited; continuing process-group cleanup")
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "Backend process %s did not stop within 60 seconds of rank-zero SIGTERM; "
                        "continuing process-group cleanup",
                        backend.pid,
                    )
            for process in reversed(processes):
                _stop(process)


def run(recipe: Recipe, smoke_test: bool = False) -> None:
    basicConfig(level=os.environ.get("AITUNE_LOG_LEVEL", "INFO"), force=True)
    if recipe.gpu_count != 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "The serve command supports a single GPU. For multiple GPUs, start the Dynamo frontend, "
            "torchrun backend, and web UI separately as shown in the README."
        )

    def interrupted(_signal: int, _frame: FrameType | None) -> NoReturn:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        _serve(recipe, smoke_test)
    except KeyboardInterrupt:
        logger.info("Shutdown requested; deployment stopped")


def command_parser() -> argparse.ArgumentParser:
    parser = argument_parser(__doc__)
    parser.add_argument(
        "--smoke_test", action="store_true", help="Run sample requests against Dynamo, then shut it down"
    )
    return parser


def main() -> None:
    args = command_parser().parse_args()
    run(load_recipe(args.recipe), smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
