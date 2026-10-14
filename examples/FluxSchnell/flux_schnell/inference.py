# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate sample images, validate RGB SSIM, and benchmark using a recipe."""

import json
import os
import threading
import time
from logging import basicConfig, getLogger

import numpy as np
import torch
import torch.distributed as dist
from skimage.metrics import structural_similarity

from .checkpoint import check_checkpoint, load_checkpoint
from .config import argument_parser, load_recipe
from .distributed import initialize, is_rank_zero, launch, shutdown, synchronize
from .model import generate, get_pipeline

logger = getLogger(__name__)


def _generate_samples(pipeline, recipe, split, variant):
    images = {}
    for sample in recipe.samples(split):
        logger.info("%s %s sample: %s (seed=%s)", variant, split, sample["id"], sample["seed"])
        output = generate(pipeline, recipe, sample)
        synchronize()
        if is_rank_zero():
            image = output.images[0]
            path = recipe.output_dir / split / variant / f"{sample['id']}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            images[sample["id"]] = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return images


def _benchmark(pipeline, recipe):
    samples = recipe.samples("benchmark")
    settings = recipe["benchmark"]
    for index in range(settings["warmup_iterations"]):
        generate(pipeline, recipe, samples[index % len(samples)])
    synchronize()
    device = torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(device)
    memory_samples = []
    done = threading.Event()

    def sample_memory():
        while not done.is_set():
            memory_samples.append(torch.cuda.memory_allocated(device) / 1024**2)
            done.wait(settings["sampling_interval_s"])

    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    latencies = []
    try:
        for index in range(settings["repetitions"]):
            synchronize()
            start = time.perf_counter()
            generate(pipeline, recipe, samples[index % len(samples)])
            synchronize()
            latencies.append(time.perf_counter() - start)
    finally:
        done.set()
        monitor.join()
    memory = {
        "rank": dist.get_rank() if dist.is_initialized() else 0,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "sampled_allocated_mib": memory_samples,
    }
    rank_memory = [memory]
    if dist.is_initialized():
        rank_memory = [None] * dist.get_world_size()
        dist.all_gather_object(rank_memory, memory)
    return {
        "samples": [sample["id"] for sample in samples],
        "warmup_iterations": settings["warmup_iterations"],
        "repetitions": settings["repetitions"],
        "sampling_interval_s": settings["sampling_interval_s"],
        "latencies_s": latencies,
        "mean_s": float(np.mean(latencies)),
        "p50_s": float(np.percentile(latencies, 50)),
        "p95_s": float(np.percentile(latencies, 95)),
        "p99_s": float(np.percentile(latencies, 99)),
        "images_per_second": len(latencies) / sum(latencies),
        "gpu_memory": rank_memory,
    }


def _validate(original, tuned, settings):
    results = []
    for sample_id, baseline in original.items():
        candidate = tuned[sample_id]
        difference = np.abs(baseline - candidate)
        score = float(structural_similarity(baseline, candidate, data_range=1.0, channel_axis=-1))
        result = {
            "id": sample_id,
            "rgb_ssim": score,
            "max_abs_diff": float(difference.max()),
            "max_rel_error": float((difference / np.maximum(np.abs(baseline), 1e-8)).max()),
            "passed": bool(np.isfinite(candidate).all() and score >= settings["threshold"]),
            "warning": score < settings["soft_threshold"],
        }
        if result["warning"]:
            logger.warning(
                "%s RGB SSIM %.4f is below soft threshold %.4f", sample_id, score, settings["soft_threshold"]
            )
        results.append(result)
    return {**settings, "samples": results, "passed": all(result["passed"] for result in results)}


def do_inference(recipe):
    manifest = check_checkpoint(recipe)
    pipeline = get_pipeline(recipe)
    validation_images, benchmarks = {}, {}
    for variant in ("original", "tuned"):
        if variant == "tuned":
            pipeline = load_checkpoint(pipeline, recipe)
        _generate_samples(pipeline, recipe, "inference", variant)
        validation_images[variant] = _generate_samples(pipeline, recipe, "validation", variant)
        benchmarks[variant] = _benchmark(pipeline, recipe)
    passed = True
    if is_rank_zero():
        validation = _validate(validation_images["original"], validation_images["tuned"], recipe["validation"])
        passed = validation["passed"]
        report = {
            "recipe": str(recipe.path),
            "model": recipe["model"],
            "gpu_count": recipe.gpu_count,
            "inference_samples": recipe.samples("inference"),
            "validation": validation,
            "benchmark": benchmarks,
            "speedup": benchmarks["original"]["mean_s"] / benchmarks["tuned"]["mean_s"],
            "tuning_seconds": manifest["tuning_seconds"],
        }
        (recipe.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        logger.info(
            "Validation passed=%s; speedup=%.3fx; report=%s",
            passed,
            report["speedup"],
            recipe.output_dir / "report.json",
        )
    if dist.is_initialized():
        status = [passed]
        dist.broadcast_object_list(status, src=0)
        passed = status[0]
    if not passed:
        raise RuntimeError("Tuned output failed the recipe's RGB SSIM threshold; see report.json")


def main():
    basicConfig(level=os.environ.get("AITUNE_LOG_LEVEL", "INFO"), force=True)
    recipe = load_recipe(argument_parser(__doc__).parse_args().recipe)
    launch(recipe, "flux_schnell.inference")
    initialize()
    try:
        do_inference(recipe)
    finally:
        shutdown()


if __name__ == "__main__":
    main()
