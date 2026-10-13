# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image requests checked against local Original Model reference PNGs."""

import base64
import io
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ..config import read_inputs
from ..evaluate import compare, require_correctness
from ..metrics import Sampler
from ..records import contract, deployment_artifact, new_run, read_json, write_json


def service_info(args, cfg):
    # Provenance and telemetry use the same results mount and host clock.
    url = urlparse(args.endpoint)
    if url.scheme != "http" or url.hostname not in ("localhost", "127.0.0.1") or url.path not in ("", "/"):
        raise ValueError("This implementation requires a local http://localhost:PORT base URL and shared results mount")
    if url.port != cfg["deployment"]["port"]:
        raise ValueError("Endpoint port differs from the selected recipe")
    service = read_json(read_json(cfg.output / f"serve-{args.variant}-latest.json")["service"])
    artifact = deployment_artifact(cfg)
    if (
        service["variant"] != args.variant
        or service["contract"] != artifact["contract"]
        or service["contract"] != contract(cfg, artifact["contract"]["source"])
        or service["artifact_sha256"] != artifact["sha256"]
        or service["seed"] != cfg["deployment"]["seed"]
    ):
        raise ValueError("Running service does not match this recipe/artifact")
    with urlopen(args.endpoint.rstrip("/") + "/v1/models", timeout=10) as response:
        names = {item["id"] for item in json.load(response)["data"]}
    if service["model_name"] not in names:
        raise ValueError("Recorded service is no longer registered; start recipe serve")
    return service


def image_request(args, cfg, service, record):
    from PIL import Image

    payload = {
        "model": service["model_name"],
        "prompt": record["prompt"],
        "n": 1,
        "size": f"{cfg['workload']['width']}x{cfg['workload']['height']}",
        "response_format": "b64_json",
    }
    request = Request(
        args.endpoint.rstrip("/") + "/v1/images/generations",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=600) as response:
        data = json.load(response)["data"]
    if len(data) != 1:
        raise ValueError("Expected one image per request")
    with Image.open(io.BytesIO(base64.b64decode(data[0]["b64_json"], validate=True))) as image:
        return image.convert("RGB")


def correctness(args, cfg):
    service = service_info(args, cfg)
    reference = require_correctness(cfg)
    records = read_inputs(cfg["inputs"]["validation"])
    if any(item["seed"] != service["seed"] for item in records):
        raise ValueError("Validation seeds must match the service seed")
    directory = new_run(cfg, f"deployment-correctness-{args.variant}")
    images = directory / "images"
    images.mkdir()
    for item in records:
        image_request(args, cfg, service, item).save(images / f"{item['id']}.png")
    reference_dir = Path(reference["original"])
    if cfg["execution"]["gpu_count"] > 1:
        reference_dir /= "rank-0"
    result = compare(cfg, reference_dir / "images", images, records)
    result.update({"service": service, "reference": reference, "run_id": directory.name})
    write_json(directory / "report.json", result)
    print(directory / "report.json")
    if not result["passed"]:
        raise RuntimeError("Deployment correctness failed")
    return result


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


def server_measurement(service, directory, start, end, count):
    directory.mkdir(parents=True, exist_ok=True)
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
    return gpu


def server_telemetry(service, directory, start, end, count):
    if "ranks" not in service:
        return {"gpu": server_measurement(service, directory, start, end, count)}
    gpus = [
        server_measurement(rank, directory / f"rank-{rank['rank']}", start, end, count) for rank in service["ranks"]
    ]
    if len({gpu["gpu_uuid"] for gpu in gpus}) != len(gpus) or any(gpu["missing_reason"] for gpu in gpus):
        raise ValueError("Require complete server telemetry for distinct GPUs")
    return {
        "gpu_count": len(gpus),
        "gpus": gpus,
        "peak_gpu_memory_gib": max(gpu["peak_gpu_memory_gib"] for gpu in gpus),
        "gpu_utilization_pct": sum(gpu["gpu_utilization_pct"] for gpu in gpus) / len(gpus),
    }


def benchmark(args, cfg):
    quality = correctness(args, cfg)
    service = quality["service"]
    records = read_inputs(cfg["inputs"]["benchmark"])
    if any(item["seed"] != service["seed"] for item in records):
        raise ValueError("Benchmark seeds must match the service seed")
    directory = new_run(cfg, f"deployment-benchmark-{args.variant}")
    dataset = directory / "aiperf-inputs.jsonl"
    dataset.write_text("".join(json.dumps({"text": item["prompt"]}) + "\n" for item in records))
    artifacts = directory / "aiperf"
    command = [
        "aiperf",
        "profile",
        "--model",
        service["model_name"],
        "--tokenizer",
        "builtin",
        "--url",
        args.endpoint,
        "--endpoint-type",
        "image_generation",
        "--input-file",
        str(dataset),
        "--custom-dataset-type",
        "single_turn",
        "--dataset-sampling-strategy",
        "sequential",
        "--extra-inputs",
        f"size:{cfg['workload']['width']}x{cfg['workload']['height']}",
        "--concurrency",
        "1",
        "--request-count",
        str(cfg["deployment"]["request_count"]),
        # AIPerf 0.13 disables warmup when omitted; an explicit zero is invalid.
        "--artifact-dir",
        str(artifacts),
    ]
    write_json(directory / "command.json", command)
    with (directory / "aiperf.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=dict(os.environ, TZ="UTC"))
    raw = read_json(artifacts / "profile_export_aiperf.json")
    if (
        raw.get("aiperf_version") != "0.13.0"
        or raw.get("is_complete") is not True
        or raw.get("was_cancelled")
        or raw.get("error_summary")
    ):
        raise ValueError("AIPerf run incomplete, incompatible, or contains request errors")
    count = metric(raw, "request_count", "requests")
    if count != cfg["deployment"]["request_count"]:
        raise ValueError("AIPerf successful request count differs from the requested count")
    start, end = timestamp(raw["start_time"]), timestamp(raw["end_time"])
    if end <= start:
        raise ValueError("Invalid AIPerf measurement window")
    telemetry = server_telemetry(service, directory, start, end, count)
    report = {
        "run_id": directory.name,
        "target": "dynamo",
        "variant": args.variant,
        "service": service,
        "correctness": quality,
        "raw_aiperf": str(artifacts),
        **telemetry,
        "throughput_images_s": count / (end - start),
        "latency_mean_ms": metric(raw, "request_latency", "ms"),
        "compilation_s": service["timing"]["compilation_s"],
        "compilation_missing_reason": service["timing"]["compilation_missing_reason"],
        "compilation_method": service["timing"].get("compilation_method"),
        "raw_compilation": service["timing"].get("raw_compilation"),
        "warmup_s": service["timing"]["warmup_s"],
        "first_use_s": service["timing"]["first_use_s"],
        "measurement_start_unix_s": start,
        "measurement_end_unix_s": end,
        "batch_size": 1,
        "concurrency": 1,
        "inputs": records,
        "measurement": "AIPerf client request window, including transport and PNG/base64 encoding; local server NVML in same window",
    }
    write_json(directory / "report.json", report)
    print(directory / "report.json")
