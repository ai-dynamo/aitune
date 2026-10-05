# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image requests checked against local Original Model reference PNGs."""

import base64
import io
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ...correctness import compare, require_correctness
from ...data import read_inputs
from ...records import artifact_metadata, contract, new_run, read_json, write_json


def service_info(args, cfg):
    # Provenance and telemetry use the same results mount and host clock.
    url = urlparse(args.endpoint)
    if url.scheme != "http" or url.hostname not in ("localhost", "127.0.0.1") or url.path not in ("", "/"):
        raise ValueError("This implementation requires a local http://localhost:PORT base URL and shared results mount")
    if url.port != cfg["deployment"]["port"]:
        raise ValueError("Endpoint port differs from the selected recipe")
    service = read_json(read_json(cfg.output / f"serve-{args.variant}-latest.json")["service"])
    artifact = artifact_metadata(cfg)
    if (service["variant"] != args.variant or service["contract"] != artifact["contract"]
            or service["contract"] != contract(cfg, artifact["contract"]["source"])
            or service["artifact_sha256"] != artifact["sha256"] or service["seed"] != cfg["deployment"]["seed"]):
        raise ValueError("Running service does not match this recipe/artifact")
    with urlopen(args.endpoint.rstrip("/") + "/v1/models", timeout=10) as response:
        names = {item["id"] for item in json.load(response)["data"]}
    if service["model_name"] not in names:
        raise ValueError("Recorded service is no longer registered; start recipe serve")
    return service


def image_request(args, cfg, service, record):
    from PIL import Image
    payload = {"model": service["model_name"], "prompt": record["prompt"], "n": 1,
               "size": f"{cfg['workload']['width']}x{cfg['workload']['height']}", "response_format": "b64_json"}
    request = Request(args.endpoint.rstrip("/") + "/v1/images/generations", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=600) as response:
        data = json.load(response)["data"]
    if len(data) != 1:
        raise ValueError("Expected one image per request")
    with Image.open(io.BytesIO(base64.b64decode(data[0]["b64_json"], validate=True))) as image:
        return image.convert("RGB")


def run(args, cfg):
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
    result = compare(cfg, Path(reference["original"]) / "images", images, records)
    result.update({"service": service, "reference": reference, "run_id": directory.name})
    write_json(directory / "report.json", result)
    print(directory / "report.json")
    if not result["passed"]:
        raise RuntimeError("Deployment correctness failed")
    return result
