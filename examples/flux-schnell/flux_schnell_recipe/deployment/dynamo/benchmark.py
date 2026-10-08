# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf image workload and server-side NVML collection."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess

from . import client
from ...data import read_inputs
from ...records import new_run, read_json, write_json
from ...telemetry import Sampler


def metric(raw, name, unit):
    entry = raw.get(name)
    if not isinstance(entry, dict) or entry.get("unit") != unit:
        raise ValueError(f"Missing or incompatible AIPerf metric {name}; expected {unit}")
    value = entry.get("avg")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid AIPerf metric {name}")
    return value


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp()


def run(args, cfg):
    quality = client.run(args, cfg)
    service = quality["service"]
    records = read_inputs(cfg["inputs"]["benchmark"])
    if any(item["seed"] != service["seed"] for item in records):
        raise ValueError("Benchmark seeds must match the service seed")
    directory = new_run(cfg, f"deployment-benchmark-{args.variant}")
    dataset = directory / "aiperf-inputs.jsonl"
    dataset.write_text("".join(json.dumps({"text": item["prompt"]}) + "\n" for item in records))
    artifacts = directory / "aiperf"
    command = ["aiperf", "profile", "--model", service["model_name"], "--tokenizer", "builtin",
               "--url", args.endpoint, "--endpoint-type", "image_generation", "--input-file", str(dataset),
               "--custom-dataset-type", "single_turn", "--dataset-sampling-strategy", "sequential",
               "--extra-inputs", f"size:{cfg['workload']['width']}x{cfg['workload']['height']}",
               "--concurrency", "1", "--request-count", str(cfg["deployment"]["request_count"]),
               # AIPerf 0.13 disables warmup when omitted; an explicit zero is invalid.
               "--artifact-dir", str(artifacts)]
    write_json(directory / "command.json", command)
    with (directory / "aiperf.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=dict(os.environ, TZ="UTC"))
    raw = read_json(artifacts / "profile_export_aiperf.json")
    if raw.get("aiperf_version") != "0.13.0" or raw.get("is_complete") is not True or raw.get("was_cancelled") or raw.get("error_summary"):
        raise ValueError("AIPerf run incomplete, incompatible, or contains request errors")
    count = metric(raw, "request_count", "requests")
    if count != cfg["deployment"]["request_count"]:
        raise ValueError("AIPerf successful request count differs from the requested count")
    start, end = timestamp(raw["start_time"]), timestamp(raw["end_time"])
    if end <= start:
        raise ValueError("Invalid AIPerf measurement window")
    gpu_path = Path(service["directory"]) / "gpu-samples.jsonl"
    # Drop a possible partial trailing line from the live append-only sampler.
    lines = gpu_path.read_text().splitlines(keepends=True)
    samples = [json.loads(line) for line in lines if line.endswith("\n")]
    if any("error" in row for row in samples):
        raise RuntimeError("Server-side GPU sampling failed")
    selected = [row for row in samples if start <= row["unix_s"] <= end]
    (directory / "gpu-samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in selected))
    sampler = Sampler(directory / "gpu-samples.jsonl", service["sampling_interval_s"])
    sampler.uuid = service["environment"]["gpu"]["uuid"]
    sampler.samples = [dict(row, monotonic_s=row["unix_s"]) for row in selected]
    gpu = sampler.summarize(start, end)
    requests = [json.loads(line) for line in (Path(service["directory"]) / "requests.jsonl").read_text().splitlines()]
    measured = [row for row in requests if start <= row["start_unix_s"] and row["end_unix_s"] <= end]
    if len(measured) != count or any(row["compiled"] for row in measured):
        raise ValueError("Server request records do not match AIPerf's window, or compilation occurred")
    write_json(directory / "server-requests.json", measured)
    report = {"run_id": directory.name, "target": "dynamo", "variant": args.variant, "service": service,
              "correctness": quality, "raw_aiperf": str(artifacts), "gpu": gpu,
              "throughput_images_s": count / (end - start), "latency_mean_ms": metric(raw, "request_latency", "ms"),
              "compilation_s": service["timing"]["compilation_s"],
              "compilation_missing_reason": service["timing"]["compilation_missing_reason"],
              "compilation_method": service["timing"].get("compilation_method"),
              "raw_compilation": service["timing"].get("raw_compilation"),
              "warmup_s": service["timing"]["warmup_s"], "first_use_s": service["timing"]["first_use_s"],
              "measurement_start_unix_s": start, "measurement_end_unix_s": end,
              "batch_size": 1, "concurrency": 1, "inputs": records,
              "measurement": "AIPerf client request window, including transport and PNG/base64 encoding; local server NVML in same window"}
    write_json(directory / "report.json", report)
    print(directory / "report.json")
