# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch isolated Python or torchrun workers, with live logs and process cleanup."""

import logging
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from .distributed import rank_directory
from .records import read_json, write_json

logger = logging.getLogger(__name__)


def worker_command(args, cfg, action, variant, directory):
    source = ["--model-id", args.model_id] if args.model_id else ["--checkpoint", str(Path(args.checkpoint).resolve())]
    launcher = [sys.executable, "-u", "-m"]
    if cfg["execution"]["gpu_count"] > 1:
        launcher = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={cfg['execution']['gpu_count']}",
            "--max-restarts=0",
            "--module",
        ]
    return [
        *launcher,
        "flux_schnell_recipe.worker",
        "--config",
        str(cfg.path),
        *source,
        "--action",
        action,
        "--variant",
        variant,
        "--run-dir",
        str(directory),
    ]


def stop_processes(processes):
    for process in reversed(processes):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in reversed(processes):
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def stream(command, log_path):
    """Retain subprocess output while showing progress, including during compilation."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.setdefault("OMP_NUM_THREADS", "1")
    started = heartbeat = time.monotonic()
    with log_path.open("w") as log, log_path.open(errors="replace") as output:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            while True:
                try:
                    code = process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    code = None
                sys.stderr.write(output.read())
                sys.stderr.flush()
                if code is not None:
                    return code
                if time.monotonic() - heartbeat >= 30:
                    logger.info("Worker running for %.0fs; log: %s", time.monotonic() - started, log_path)
                    heartbeat = time.monotonic()
        finally:
            stop_processes([process])


def launch(args, cfg, action, variant, directory, reference=None):
    directory.mkdir(parents=True, exist_ok=False)
    command = worker_command(args, cfg, action, variant, directory)
    if reference is not None:
        command += ["--reference", str(reference)]
    write_json(directory / "command.json", command)
    log_path = directory / "torchrun.log" if cfg["execution"]["gpu_count"] > 1 else directory.parent / f"{variant}.log"
    logger.info("Starting %s/%s; log: %s", action, variant, log_path)
    code = stream(command, log_path)
    if code:
        errors = [
            f"Rank {rank} worker traceback ({path}):\n{read_json(path)['traceback']}"
            for rank in range(cfg["execution"]["gpu_count"])
            if (path := rank_directory(cfg, directory, rank) / "error.json").exists()
        ]
        with log_path.open(errors="replace") as log:
            tail = "".join(deque(log, maxlen=30))
        details = "\n".join(errors)
        raise RuntimeError(f"{action}/{variant} exited with status {code}. Full log: {log_path}\n{details}\n{tail}")
    return [
        read_json(rank_directory(cfg, directory, rank) / "report.json") for rank in range(cfg["execution"]["gpu_count"])
    ]
