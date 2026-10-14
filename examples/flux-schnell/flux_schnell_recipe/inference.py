# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate baseline and tuned images, check SSIM, and measure steady-state inference."""

from __future__ import annotations

import logging
import time
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import (
    Config,
    InputRecord,
    argument_parser,
    configure,
    configure_cache,
    rank_config,
    read_inputs,
    read_json,
    write_json,
)
from .distributed import collect, initialize, rank_directory, shutdown
from .metrics import CompilationRecorder, Sampler, compilation_counters, interval_seconds, phase_duration, prepare
from .model import contract, generate, load_model, load_tuned, save_images, synchronize

if TYPE_CHECKING:
    from diffusers import FluxPipeline
    from PIL.Image import Image

logger = logging.getLogger(__name__)


def score_images(reference: Image, actual: Image) -> float:
    import numpy as np
    from skimage.metrics import structural_similarity

    a, b = np.asarray(reference.convert("RGB")), np.asarray(actual.convert("RGB"))
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError(f"Image shapes differ: {a.shape} and {b.shape}")
    score = float(structural_similarity(a, b, data_range=255, channel_axis=-1))
    if not np.isfinite(score):
        raise ValueError("SSIM produced a nonfinite score")
    return score


def compare(cfg: Config, reference: str | Path, actual: str | Path, records: Sequence[InputRecord]) -> dict[str, Any]:
    from PIL import Image

    threshold = cfg["validation"]["threshold"]
    soft_threshold = cfg["validation"].get("soft_threshold")
    scores = []
    for item in records:
        with (
            Image.open(Path(reference) / f"{item['id']}.png") as a,
            Image.open(Path(actual) / f"{item['id']}.png") as b,
        ):
            expected_size = (cfg["workload"]["width"], cfg["workload"]["height"])
            if a.size != expected_size or b.size != expected_size:
                raise ValueError("Image dimensions differ from the recipe")
            score = score_images(a, b)
        passed = score >= threshold
        soft_passed = score >= soft_threshold if soft_threshold is not None else None
        if passed and soft_passed is False:
            logger.warning(
                "Image %s: SSIM %.6f is below soft threshold %.6f (hard threshold %.6f passed)",
                item["id"],
                score,
                soft_threshold,
                threshold,
            )
        scores.append({"id": item["id"], "ssim": score, "passed": passed, "soft_passed": soft_passed})
    return {
        "metric": "rgb_ssim",
        "threshold": threshold,
        "soft_threshold": soft_threshold,
        "passed": all(s["passed"] for s in scores),
        "soft_passed": all(s["soft_passed"] for s in scores) if soft_threshold is not None else None,
        "scores": scores,
    }


def measure(
    pipe: FluxPipeline, cfg: Config, records: Sequence[InputRecord], directory: Path, recorder: CompilationRecorder
) -> dict[str, Any]:
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


def benchmark(
    pipe: FluxPipeline,
    cfg: Config,
    records: Sequence[InputRecord],
    directory: Path,
    recorder: CompilationRecorder,
    timing: dict[str, Any],
) -> dict[str, Any]:
    ranks = collect({**timing, **measure(pipe, cfg, records, directory, recorder)})
    compiler_records = collect(recorder.snapshot())
    duration = phase_duration(ranks, "measurement")
    count = cfg["benchmark"]["repetitions"]
    timings = collect(read_json(directory / "timings.json"))
    latencies = [
        max(row["end_monotonic_s"] for row in rows) - min(row["start_monotonic_s"] for row in rows)
        for rows in zip(*timings, strict=True)
    ]
    return {
        "images": count,
        "throughput_images_s": count / duration,
        "latency_mean_ms": sum(latencies) / count * 1000,
        "first_use_s": phase_duration(ranks, "first_use"),
        "warmup_s": phase_duration(ranks, "warmup"),
        "compilation_s": interval_seconds([row for rows in compiler_records for row in rows]),
        "gpus": [rank["gpu"] for rank in ranks],
        "ranks": ranks,
    }


def run(args: Namespace, cfg: Config, rank: int) -> None:
    directory = cfg.output / args.mode
    local_cfg = rank_config(cfg, rank)
    phase = {"infer": "inference", "correctness": "validation", "benchmark": "benchmark"}[args.mode]
    records = read_inputs(cfg["inputs"][phase])
    validation = read_inputs(cfg["inputs"]["validation"])
    variants = ("original", "aitune") if args.variant == "both" else (args.variant,)
    pipe = load_model(args, local_cfg)
    results: dict[str, Any] = {}
    for variant in variants:
        output = rank_directory(cfg, directory / variant, rank)
        output.mkdir(parents=True, exist_ok=True)
        recorder = CompilationRecorder(output)
        if variant == "aitune":
            pipe = load_tuned(pipe, args, local_cfg)
        if args.mode == "benchmark":
            timing = prepare(pipe, cfg, records + validation, recorder)
            save_images(pipe, cfg, validation, directory / variant / "images")
        else:
            timing = {}
            save_images(pipe, cfg, records, directory / variant / "images")
        synchronize()
        if variant == "aitune" and args.mode != "infer":
            quality = compare(cfg, directory / "original" / "images", directory / "aitune" / "images", validation)
            if rank == 0:
                write_json(directory / "correctness.json", quality)
            if not all(collect(quality["passed"])):
                raise RuntimeError("Image correctness failed; inspect the saved PNGs and SSIM scores")
            results["correctness"] = quality
        if args.mode == "benchmark":
            results[variant] = benchmark(pipe, cfg, records, output, recorder, timing)
    if args.mode == "benchmark":
        results["speedup"] = results["aitune"]["throughput_images_s"] / results["original"]["throughput_images_s"]
    if rank == 0:
        write_json(
            directory / "report.json",
            {"recipe": cfg.data, "contract": contract(args, cfg), "inputs": records, **results},
        )
        print(directory / "report.json", flush=True)


def main() -> None:
    parser = argument_parser(__doc__, source=True)
    parser.add_argument("--mode", choices=("infer", "correctness", "benchmark"), default="infer")
    parser.add_argument("--variant", choices=("original", "aitune", "both"), default="both")
    args = parser.parse_args()
    if args.mode != "infer" and args.variant != "both":
        parser.error("Correctness and benchmarking compare both variants; omit --variant")
    cfg = configure(args)
    configure_cache(cfg)
    try:
        rank = initialize(cfg)
        if rank == 0:
            directory = cfg.output / args.mode
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "report.json").unlink(missing_ok=True)
            (directory / "correctness.json").unlink(missing_ok=True)
        run(args, cfg, rank)
    finally:
        shutdown()


if __name__ == "__main__":
    main()
