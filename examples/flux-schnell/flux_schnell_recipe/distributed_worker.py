# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One rank in a fresh two-GPU tuning, correctness, inference, or benchmark job."""

import argparse
import logging
from pathlib import Path
import time
import traceback

from .config import load_config
from .distributed import collect, initialize, rank_config, shutdown
from .records import configure_cache, contract, environment, source_identity, write_json

logger = logging.getLogger(__name__)


def execute(args, cfg, rank, directory):
    from .compilation import CompilationRecorder
    from .data import read_inputs, save_images
    from .model import generate, load_model, synchronize
    from .worker import compilation_counters, prepare
    if args.action == "tune":
        from .tune import run
        run(args, cfg, run_dir=directory)
        return
    recorder = CompilationRecorder(directory)
    source = source_identity(args, cfg)
    pipe = load_model(args, cfg, args.variant, source)
    report = {"rank": rank, "variant": args.variant, "contract": contract(cfg, source),
              "environment": environment(), "directory": str(directory)}
    if args.action != "benchmark":
        phase = "inference" if args.action == "infer" else "validation"
        records = read_inputs(cfg["inputs"][phase])
        save_images(pipe, cfg, records, directory / "images")
        synchronize()
        report["inputs"] = records
    else:
        from .correctness import compare
        from .telemetry import Sampler
        records = read_inputs(cfg["inputs"]["benchmark"])
        validation = read_inputs(cfg["inputs"]["validation"])
        report.update(prepare(pipe, cfg, records + validation, recorder))
        save_images(pipe, cfg, validation, directory / "images")
        quality = compare(cfg, Path(args.reference) / f"rank-{rank}" / "images",
                          directory / "images", validation)
        write_json(directory / "correctness.json", quality)
        if not all(collect(quality["passed"])):
            raise RuntimeError("At least one fresh benchmark rank failed correctness")
        report["correctness"] = quality
        sampler = Sampler(directory / "gpu-samples.jsonl", cfg["benchmark"]["sampling_interval_s"]).start()
        before, compilation_records = compilation_counters(), recorder.snapshot()
        timings = []
        synchronize()
        start = time.perf_counter()
        try:
            for index in range(cfg["benchmark"]["repetitions"]):
                item = records[index % len(records)]
                t0 = time.perf_counter()
                generate(pipe, cfg, item)
                synchronize()
                timings.append({"id": item["id"], "start_monotonic_s": t0,
                                "end_monotonic_s": time.perf_counter()})
        finally:
            end = time.perf_counter()
            sampler.stop()
        windows = collect((start, end))
        lo, hi = min(w[0] for w in windows), max(w[1] for w in windows)
        if any(collect(compilation_counters() != before or recorder.snapshot() != compilation_records)):
            raise RuntimeError("Compilation occurred in the measurement window")
        gpu = sampler.summarize(lo, hi)
        if any(collect(gpu["missing_reason"] is not None)):
            raise RuntimeError("GPU telemetry is incomplete on at least one rank")
        write_json(directory / "timings.json", timings)
        report.update({"gpu": gpu, "measurement_start_monotonic_s": lo, "measurement_end_monotonic_s": hi,
                       "raw_timings": str(directory / "timings.json"), "batch_size": 1, "concurrency": 1,
                       "workload_inputs": records})
    write_json(directory / "report.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-id")
    source.add_argument("--checkpoint")
    parser.add_argument("--action", required=True, choices=("tune", "correctness", "benchmark", "infer"))
    parser.add_argument("--variant", required=True, choices=("original", "aitune"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reference")
    args = parser.parse_args()
    cfg = load_config(args.config)
    # Configure caches before importing Torch/AITune; each torchrun child has its own files.
    import os
    rank = int(os.environ["RANK"])
    directory = Path(args.run_dir) / f"rank-{rank}"
    directory.mkdir(parents=True, exist_ok=False)
    configure_cache(directory)
    logging.basicConfig(level=logging.INFO,
                        format=f"%(asctime)s %(levelname)s [{args.action}/{args.variant}/rank-{rank}] %(message)s",
                        datefmt="%H:%M:%S")
    logger.info("Initializing distributed worker; results: %s", directory)
    try:
        initialize(cfg)
        local_cfg = rank_config(cfg, rank)
        write_json(directory / "resolved-config.json", local_cfg.data)
        logger.info("Starting %s", args.action)
        execute(args, local_cfg, rank, directory)
        logger.info("Finished %s; report: %s", args.action, directory / "report.json")
    except Exception:
        write_json(directory / "error.json", {"rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        shutdown()


if __name__ == "__main__":
    main()
