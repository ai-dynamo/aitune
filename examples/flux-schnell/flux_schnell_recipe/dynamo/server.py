# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Own the Dynamo frontend and worker; readiness and process-group cleanup."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

from ..records import contract, deployment_artifact, new_run, read_json, source_identity, write_json
from ..runtime import stop_processes, worker_command


def publish_distributed_service(cfg, directory, artifact):
    paths = [directory / f"rank-{rank}" / "service.json" for rank in range(cfg["execution"]["gpu_count"])]
    if not all(path.is_file() for path in paths):
        return
    ranks = [read_json(path) for path in paths]
    for rank, service in enumerate(ranks):
        if (
            service["rank"] != rank
            or service["contract"] != artifact["contract"]
            or service["artifact_sha256"] != artifact["sha256"][rank]
            or service["model_name"] != ranks[0]["model_name"]
        ):
            raise ValueError("Rank service records differ from the tuning manifest")
    from ..metrics import interval_seconds, phase_duration

    timing = dict(ranks[0]["timing"])
    timings = [service["timing"] for service in ranks]
    raw = [item["raw_compilation"] for item in timings]
    timing.update({
        "first_use_s": phase_duration(timings, "first_use"),
        "warmup_s": phase_duration(timings, "warmup"),
        "compilation_s": interval_seconds([r for path in raw for r in read_json(path)["records"]]),
        "raw_compilation": raw,
        "compilation_method": "Union of compiler intervals across ranks; distributed first-use and warmup spans",
    })
    service = {key: value for key, value in ranks[0].items() if key != "rank"}
    service.update({
        "directory": str(directory),
        "gpu_count": len(ranks),
        "ranks": ranks,
        "artifact_sha256": artifact["sha256"],
        "timing": timing,
    })
    write_json(directory / "service.json", service)


def registered(directory, port):
    if not (directory / "service.json").exists():
        return False
    service = read_json(directory / "service.json")
    try:
        with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
            models = json.load(response)
        return service["model_name"] in {m["id"] for m in models["data"]}
    except (OSError, ValueError, KeyError):
        return False


def wait_for_ready(processes, cfg, directory, artifact):
    deadline = time.monotonic() + cfg["deployment"]["readiness_timeout_s"]
    while True:
        if any(p.poll() is not None for p in processes):
            raise RuntimeError(f"Dynamo process exited; inspect {directory}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"Dynamo readiness timed out; inspect {directory}")
        if cfg["execution"]["gpu_count"] > 1 and not (directory / "service.json").exists():
            publish_distributed_service(cfg, directory, artifact)
        if registered(directory, cfg["deployment"]["port"]):
            return
        time.sleep(1)


def run(args, cfg):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or "RANK" in os.environ:
        raise ValueError("Run recipe serve once; it launches its own workers. Do not wrap it in torchrun")
    artifact = deployment_artifact(cfg)
    if artifact["contract"] != contract(cfg, source_identity(args, cfg)):
        raise ValueError("Service source/configuration differs from the artifact")
    port = cfg["deployment"]["port"]
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("0.0.0.0", port))
    directory = new_run(cfg, f"serve-{args.variant}")
    env = dict(
        os.environ,
        DYN_DISCOVERY_BACKEND="file",
        DYN_EVENT_PLANE="zmq",
        DYN_REQUEST_PLANE="tcp",
        DYN_ROUTER_USE_KV_EVENTS="false",
        TZ="UTC",
    )
    worker = worker_command(args, cfg, "serve", args.variant, directory)
    env.setdefault("OMP_NUM_THREADS", "1")
    env["PYTHONUNBUFFERED"] = "1"
    commands = [[sys.executable, "-m", "dynamo.frontend", "--http-port", str(port)], worker]
    write_json(directory / "commands.json", commands)
    print(f"Starting Dynamo; logs: {directory / 'frontend.log'} and {directory / 'worker.log'}", flush=True)
    processes, logs = [], []
    old_handler = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        for name, command in zip(("frontend", "worker"), commands, strict=True):
            log = (directory / f"{name}.log").open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            )
        wait_for_ready(processes, cfg, directory, artifact)
        write_json(cfg.output / f"serve-{args.variant}-latest.json", {"service": str(directory / "service.json")})
        print(f"Ready at http://localhost:{port}; records: {directory}; stop with Ctrl-C", flush=True)
        while all(p.poll() is None for p in processes):
            time.sleep(1)
        raise RuntimeError(f"Dynamo exited unexpectedly; inspect {directory}")
    except KeyboardInterrupt:
        pass
    finally:
        stop_processes(processes)
        for log in logs:
            log.close()
        signal.signal(signal.SIGTERM, old_handler)
