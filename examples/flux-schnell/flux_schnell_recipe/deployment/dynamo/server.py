# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Own the Dynamo frontend and worker; readiness and process-group cleanup."""

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

from ...records import new_run, read_json, write_json


def run(args, cfg):
    port = cfg["deployment"]["port"]
    with socket.socket() as probe:
        probe.bind(("0.0.0.0", port))
    directory = new_run(cfg, f"serve-{args.variant}")
    env = dict(os.environ, DYN_DISCOVERY_BACKEND="file", DYN_EVENT_PLANE="zmq", DYN_REQUEST_PLANE="tcp",
               DYN_ROUTER_USE_KV_EVENTS="false", TZ="UTC")
    source = ["--model-id", args.model_id] if args.model_id else ["--checkpoint", str(Path(args.checkpoint).resolve())]
    worker = [sys.executable, "-m", "flux_schnell_recipe.deployment.dynamo.backend", "--config", str(cfg.path),
              *source, "--variant", args.variant, "--run-dir", str(directory)]
    commands = [[sys.executable, "-m", "dynamo.frontend", "--http-port", str(port)], worker]
    processes, logs = [], []
    old_handler = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        for name, command in zip(("frontend", "worker"), commands):
            log = (directory / f"{name}.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True))
        deadline = time.monotonic() + cfg["deployment"]["readiness_timeout_s"]
        while True:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError(f"Dynamo process exited; inspect {directory}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"Dynamo readiness timed out; inspect {directory}")
            if (directory / "service.json").exists():
                service = read_json(directory / "service.json")
                try:
                    with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
                        models = json.load(response)
                    if service["model_name"] in {m["id"] for m in models["data"]}:
                        break
                except (OSError, ValueError, KeyError):
                    pass
            time.sleep(1)
        write_json(cfg.output / f"serve-{args.variant}-latest.json", {"service": str(directory / "service.json")})
        print(f"Ready at http://localhost:{port}; records: {directory}; stop with Ctrl-C", flush=True)
        while all(p.poll() is None for p in processes):
            time.sleep(1)
        raise RuntimeError(f"Dynamo exited unexpectedly; inspect {directory}")
    except KeyboardInterrupt:
        pass
    finally:
        for p in reversed(processes):
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in reversed(processes):
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        for log in logs:
            log.close()
        signal.signal(signal.SIGTERM, old_handler)
