# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamo's image protocol, backed by the same pipeline as Python commands."""

import argparse
import io
import json
from pathlib import Path
import threading
import time

from ...config import load_config
from ...data import read_inputs
from ...records import artifact_metadata, configure_cache, contract, environment, source_identity, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-id")
    source.add_argument("--checkpoint")
    parser.add_argument("--variant", required=True, choices=("original", "aitune"))
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    directory = Path(args.run_dir)
    configure_cache(directory)
    import aitune.dynamo as dyn
    from ...model import load_model, generate, synchronize
    from ...telemetry import Sampler
    from ...worker import prepare, compilation_counters
    from ...compilation import CompilationRecorder
    identity = source_identity(args, cfg)
    artifact = artifact_metadata(cfg)
    if contract(cfg, identity) != artifact["contract"]:
        raise ValueError("Service source/configuration differs from the artifact")
    records = read_inputs(cfg["inputs"]["validation"]) + read_inputs(cfg["inputs"]["benchmark"])
    if any(r["seed"] != cfg["deployment"]["seed"] for r in records):
        raise ValueError("Dynamo's image protocol uses one configured seed; validation/benchmark seeds must match it")
    compilation = CompilationRecorder(directory)
    pipe = load_model(args, cfg, args.variant, identity)
    name = f"flux-schnell-{args.variant}-{directory.name}"
    lock = threading.Lock()
    sampler = None

    def warmup():
        nonlocal sampler
        timing = prepare(pipe, cfg, records, compilation)
        sampler = Sampler(directory / "gpu-samples.jsonl", cfg["benchmark"]["sampling_interval_s"], stream=True).start()
        write_json(directory / "service.json", {"run_id": directory.name, "model_name": name,
            "variant": args.variant, "contract": contract(cfg, identity), "artifact_sha256": artifact["sha256"],
            "environment": environment(), "timing": timing, "directory": str(directory),
            "seed": cfg["deployment"]["seed"], "sampling_interval_s": cfg["benchmark"]["sampling_interval_s"]})

    def mapping(request):
        size = f"{cfg['workload']['width']}x{cfg['workload']['height']}"
        if request.size not in (None, size) or getattr(request, "n", 1) not in (None, 1):
            raise ValueError(f"This artifact accepts n=1 and size={size}")
        return {"prompt": request.prompt}

    def serve(prompt):
        with lock:
            before = compilation_counters()
            compile_records = compilation.snapshot()
            start = time.time()
            image = generate(pipe, cfg, {"prompt": prompt, "seed": cfg["deployment"]["seed"]})
            synchronize()
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            compiled = compilation_counters() != before or compilation.snapshot() != compile_records
            with (directory / "requests.jsonl").open("a") as log:
                log.write(json.dumps({"start_unix_s": start, "end_unix_s": time.time(), "compiled": compiled}) + "\n")
            if compiled:
                raise RuntimeError("Unexpected runtime compilation after service warmup; restart and inspect workload")
            return buffer.getvalue()

    try:
        dyn.dynamo_worker(serve, dyn.DynamoWorkerConfig(type="image", model_path=args.model_id or args.checkpoint,
            model_name=name, namespace="flux_" + directory.name.replace("-", "_"), mapping=mapping), warmup=warmup)
    finally:
        if sampler:
            sampler.stop()


if __name__ == "__main__":
    main()
