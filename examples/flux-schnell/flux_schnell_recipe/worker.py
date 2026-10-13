# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The same tuning, inference, benchmark, or serving worker on one or two GPUs."""

import argparse
import logging
import os
import time
import traceback
from pathlib import Path

from .config import load_config, rank_config, read_inputs
from .distributed import collect, initialize, rank_directory, shutdown
from .metrics import CompilationRecorder, Sampler, compilation_counters, prepare
from .model import generate, load_model, save_images, synchronize
from .records import configure_cache, contract, environment, source_identity, write_json

logger = logging.getLogger(__name__)


def measure(pipe, cfg, records, directory, recorder):
    sampler = Sampler(directory / "gpu-samples.jsonl", cfg["benchmark"]["sampling_interval_s"]).start()
    before, compilation_records = compilation_counters(), recorder.snapshot()
    timings = []
    synchronize()
    start, unix_start = time.perf_counter(), time.time()
    try:
        for index in range(cfg["benchmark"]["repetitions"]):
            item = records[index % len(records)]
            t0 = time.perf_counter()
            generate(pipe, cfg, item)
            synchronize()
            end = time.perf_counter()
            timings.append({"id": item["id"], "start_monotonic_s": t0, "end_monotonic_s": end, "elapsed_s": end - t0})
    finally:
        end = time.perf_counter()
        sampler.stop()
    windows = collect((start, end))
    lo, hi = min(w[0] for w in windows), max(w[1] for w in windows)
    if any(collect(compilation_counters() != before or recorder.snapshot() != compilation_records)):
        raise RuntimeError("Compilation occurred in the measurement window")
    gpu = sampler.summarize(lo, hi)
    if cfg["execution"]["gpu_count"] > 1 and any(collect(gpu["missing_reason"] is not None)):
        raise RuntimeError("GPU telemetry is incomplete on at least one rank")
    write_json(directory / "timings.json", timings)
    return {
        "gpu": gpu,
        "measurement_start_monotonic_s": lo,
        "measurement_end_monotonic_s": hi,
        "measurement_start_unix_s": unix_start,
        "measurement_duration_s": hi - lo,
        "throughput_images_s": len(timings) / (hi - lo),
        "latency_mean_ms": sum(t["elapsed_s"] for t in timings) / len(timings) * 1000,
        "raw_timings": str(directory / "timings.json"),
        "batch_size": 1,
        "concurrency": 1,
        "workload_inputs": records,
    }


def execute(args, cfg, rank, directory):
    if args.action == "tune":
        from .tune import run

        return run(args, cfg, run_dir=directory)
    if args.action == "serve":
        from .dynamo.backend import execute as serve

        return serve(args, cfg, directory, rank)
    recorder = CompilationRecorder(directory)
    source = source_identity(args, cfg)
    pipe = load_model(args, cfg, args.variant, source)
    report = {
        "rank": rank,
        "variant": args.variant,
        "contract": contract(cfg, source),
        "run_id": directory.parent.name,
        "environment": environment(),
        "directory": str(directory),
    }
    phase = {"infer": "inference", "correctness": "validation", "benchmark": "benchmark"}[args.action]
    records = read_inputs(cfg["inputs"][phase])
    if args.action == "benchmark":
        from .evaluate import compare

        validation = read_inputs(cfg["inputs"]["validation"])
        report.update(prepare(pipe, cfg, records + validation, recorder))
        save_images(pipe, cfg, validation, directory / "images")
        quality = compare(
            cfg, rank_directory(cfg, Path(args.reference), rank) / "images", directory / "images", validation
        )
        write_json(directory / "correctness.json", quality)
        if not all(collect(quality["passed"])):
            raise RuntimeError("Fresh benchmark model failed correctness on at least one rank")
        report["correctness"] = quality
        logger.info("Measuring %d images", cfg["benchmark"]["repetitions"])
        report.update(measure(pipe, cfg, records, directory, recorder))
    else:
        save_images(pipe, cfg, records, directory / "images")
        synchronize()
        report["inputs"] = records
    write_json(directory / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-id")
    source.add_argument("--checkpoint")
    parser.add_argument("--action", required=True, choices=("tune", "correctness", "benchmark", "infer", "serve"))
    parser.add_argument("--variant", required=True, choices=("original", "aitune"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reference")
    args = parser.parse_args()
    if args.action == "benchmark" and args.reference is None:
        parser.error("benchmark requires --reference from a completed correctness run")
    cfg = load_config(args.config)
    rank = int(os.environ.get("RANK", "0"))
    directory = rank_directory(cfg, Path(args.run_dir), rank)
    directory.mkdir(parents=True, exist_ok=cfg["execution"]["gpu_count"] == 1)
    configure_cache(directory)  # Before Torch/AITune imports; each rank owns its caches.
    logging.basicConfig(
        level=logging.INFO, format=f"%(asctime)s %(levelname)s [{args.action}/{args.variant}/rank-{rank}] %(message)s"
    )
    try:
        initialize(cfg)
        local_cfg = rank_config(cfg, rank)
        write_json(directory / "resolved-config.json", local_cfg.data)
        logger.info("Starting %s; results: %s", args.action, directory)
        execute(args, local_cfg, rank, directory)
    except Exception:
        write_json(directory / "error.json", {"rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        shutdown()


if __name__ == "__main__":
    main()
