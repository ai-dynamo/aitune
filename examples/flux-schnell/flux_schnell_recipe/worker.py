# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A fresh interpreter owns exactly one variant and its CUDA allocations."""

import argparse
from pathlib import Path
import time

from .config import load_config
from .data import read_inputs, save_images
from .records import configure_cache, contract, environment, source_identity, write_json


def compilation_counters():
    from torch._dynamo.utils import counters
    from torch._inductor import metrics
    return {"dynamo_unique_graphs": counters["stats"]["unique_graphs"],
            "inductor_generated_kernels": metrics.generated_kernel_count}


def prepare(pipe, cfg, records):
    from .model import generate, synchronize
    synchronize()
    start = time.perf_counter()
    for item in records:
        generate(pipe, cfg, item)
    synchronize()
    first_use_s = time.perf_counter() - start
    start = time.perf_counter()
    before = compilation_counters()
    for _ in range(cfg["benchmark"]["warmup_iterations"]):
        for item in records:
            generate(pipe, cfg, item)
    synchronize()
    warmup_s = time.perf_counter() - start
    if compilation_counters() != before:
        raise RuntimeError("Compilation continued during warmup; inspect logs before accepting measurements")
    return {"first_use_s": first_use_s, "warmup_s": warmup_s, "compilation_s": None,
            "compilation_missing_reason": "First-use includes generation and lazy compilation; pure compiler wall time is unavailable",
            "first_use_policy": "One unmeasured pass over every input before post-compilation warmup",
            "cache_state": "new per-process compiler caches; artifact retains tuning-time state"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-id")
    src.add_argument("--checkpoint")
    parser.add_argument("--action", required=True, choices=("correctness", "benchmark", "infer"))
    parser.add_argument("--variant", required=True, choices=("original", "aitune"))
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    directory = Path(args.run_dir) / args.variant
    directory.mkdir()
    configure_cache(directory)
    from .model import generate, load_model, synchronize
    source = source_identity(args, cfg)
    pipe = load_model(args, cfg, args.variant, source)
    report = {"variant": args.variant, "run_id": Path(args.run_dir).name, "contract": contract(cfg, source),
              "environment": environment()}
    if args.action != "benchmark":
        phase = "validation" if args.action == "correctness" else "inference"
        records = read_inputs(cfg["inputs"][phase])
        save_images(pipe, cfg, records, directory / "images")
        report["inputs"] = records
    else:
        from .correctness import compare, require_correctness
        from .telemetry import Sampler
        evidence = require_correctness(cfg)
        validation = read_inputs(cfg["inputs"]["validation"])
        records = read_inputs(cfg["inputs"]["benchmark"])
        report.update(prepare(pipe, cfg, records + validation))
        save_images(pipe, cfg, validation, directory / "images")
        quality = compare(cfg, Path(evidence["original"]) / "images", directory / "images", validation)
        write_json(directory / "correctness.json", quality)
        if not quality["passed"]:
            raise RuntimeError("Fresh benchmark model failed correctness")
        report["correctness"] = quality
        sampler = Sampler(directory / "gpu-samples.jsonl", cfg["benchmark"]["sampling_interval_s"]).start()
        timings = []
        before = compilation_counters()
        synchronize()
        start = time.perf_counter()
        unix_start = time.time()
        try:
            for index in range(cfg["benchmark"]["repetitions"]):
                item = records[index % len(records)]
                t0 = time.perf_counter()
                generate(pipe, cfg, item)
                synchronize()
                timings.append({"id": item["id"], "elapsed_s": time.perf_counter() - t0})
        finally:
            end = time.perf_counter()
            sampler.stop()
        write_json(directory / "timings.json", timings)
        if compilation_counters() != before:
            raise RuntimeError("Compilation occurred in the measured window; report rejected")
        report.update({"throughput_images_s": len(timings) / (end - start),
                       "latency_mean_ms": sum(t["elapsed_s"] for t in timings) / len(timings) * 1000,
                       "gpu": sampler.summarize(start, end), "measurement_start_unix_s": unix_start,
                       "measurement_duration_s": end - start, "batch_size": 1, "concurrency": 1,
                       "workload_inputs": records, "raw_timings": str(directory / "timings.json")})
    write_json(directory / "report.json", report)


if __name__ == "__main__":
    main()
