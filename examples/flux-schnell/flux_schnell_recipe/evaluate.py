# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate images, benchmark fresh models, and run inference on either GPU layout."""

from __future__ import annotations

import logging
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import Config, InputRecord, read_inputs
from .distributed import rank_directory
from .metrics import interval_seconds, phase_duration
from .records import contract, deployment_artifact, file_sha, new_run, read_json, source_identity, write_json
from .runtime import launch

if TYPE_CHECKING:
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


def require_correctness(cfg: Config, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(read_json(cfg.output / "correctness-latest.json")["report"])
    result = read_json(path if path.is_absolute() else cfg.output / path)
    metadata = metadata or deployment_artifact(cfg)
    thresholds = (
        [rank["threshold"] for rank in result["ranks"]] if cfg["execution"]["gpu_count"] > 1 else [result["threshold"]]
    )
    if (
        not result["passed"]
        or result["contract"] != metadata["contract"]
        or result["artifact_sha256"] != metadata["sha256"]
        or result["validation_inputs_sha256"] != file_sha(cfg["inputs"]["validation"])
        or thresholds != [cfg["validation"]["threshold"]] * cfg["execution"]["gpu_count"]
    ):
        raise ValueError("Correctness evidence is stale; run recipe correctness with this configuration")
    return result


def correctness(args: Namespace, cfg: Config) -> dict[str, Any]:
    artifact = deployment_artifact(cfg)
    if artifact["contract"] != contract(cfg, source_identity(args, cfg)):
        raise ValueError("Artifact contract differs from requested source/configuration")
    directory = new_run(cfg, "correctness")
    variants = {
        variant: launch(args, cfg, "correctness", variant, directory / variant) for variant in ("original", "aitune")
    }
    inputs = read_inputs(cfg["inputs"]["validation"])
    scores = [
        compare(
            cfg,
            rank_directory(cfg, directory / "original", rank) / "images",
            rank_directory(cfg, directory / "aitune", rank) / "images",
            inputs,
        )
        for rank in range(cfg["execution"]["gpu_count"])
    ]
    quality = scores[0] if len(scores) == 1 else {"passed": all(s["passed"] for s in scores), "ranks": scores}
    result = {
        **quality,
        "run_id": directory.name,
        "contract": artifact["contract"],
        "artifact_sha256": artifact["sha256"],
        "variants": variants,
        "validation_inputs_sha256": file_sha(cfg["inputs"]["validation"]),
        "original": str(directory / "original"),
        "aitune": str(directory / "aitune"),
    }
    write_json(directory / "report.json", result)
    print(directory / "report.json", flush=True)
    if not result["passed"]:
        raise RuntimeError("Image correctness failed; inspect the saved images and scores")
    write_json(cfg.output / "correctness-latest.json", {"report": str(directory / "report.json")})
    return result


def aggregate(ranks: list[dict[str, Any]], directory: Path, cfg: Config) -> dict[str, Any]:
    if len(ranks) == 1:
        return ranks[0]
    if [r["rank"] for r in ranks] != list(range(cfg["execution"]["gpu_count"])):
        raise ValueError("Incomplete rank reports")
    if any(r["contract"] != ranks[0]["contract"] or r["workload_inputs"] != ranks[0]["workload_inputs"] for r in ranks):
        raise ValueError("Rank workloads differ")
    timings = [read_json(Path(r["directory"]) / "timings.json") for r in ranks]
    count = cfg["benchmark"]["repetitions"]
    if any(len(rows) != count for rows in timings):
        raise ValueError("Incomplete distributed measurement")
    joined = []
    for rows in zip(*timings, strict=True):
        elapsed = max(r["end_monotonic_s"] for r in rows) - min(r["start_monotonic_s"] for r in rows)
        if len({r["id"] for r in rows}) != 1 or elapsed <= 0:
            raise ValueError("Ranks processed different inputs or invalid timing intervals")
        joined.append({"id": rows[0]["id"], "elapsed_s": elapsed})
    duration = phase_duration(ranks, "measurement")
    gpus = [r["gpu"] for r in ranks]
    if len({gpu["gpu_uuid"] for gpu in gpus}) != len(ranks) or any(gpu["missing_reason"] for gpu in gpus):
        raise ValueError("Require complete telemetry for distinct GPUs")
    compiler_records = [
        record for rank in ranks for record in read_json(Path(rank["directory"]) / "compilation.json")["records"]
    ]
    write_json(directory / "timings.json", joined)
    result = {
        "variant": ranks[0]["variant"],
        "images": count,
        "batch_size": 1,
        "concurrency": 1,
        "throughput_images_s": count / duration,
        "latency_mean_ms": sum(row["elapsed_s"] for row in joined) / count * 1000,
        "peak_gpu_memory_gib": max(gpu["peak_gpu_memory_gib"] for gpu in gpus),
        "gpu_utilization_pct": sum(gpu["gpu_utilization_pct"] for gpu in gpus) / len(gpus),
        "compilation_s": interval_seconds(compiler_records),
        "warmup_s": phase_duration(ranks, "warmup"),
        "first_use_s": phase_duration(ranks, "first_use"),
        "measurement_duration_s": duration,
        "gpus": gpus,
        "ranks": ranks,
        "correctness": {
            "passed": all(r["correctness"]["passed"] for r in ranks),
            "ranks": [r["correctness"] for r in ranks],
        },
    }
    write_json(directory / "report.json", result)
    return result


def benchmark(args: Namespace, cfg: Config) -> None:
    quality = correctness(args, cfg)
    directory = new_run(cfg, "benchmark")
    variants = {}
    for variant in ("original", "aitune"):
        ranks = launch(args, cfg, "benchmark", variant, directory / variant, Path(quality["original"]))
        variants[variant] = aggregate(ranks, directory / variant, cfg)
    report = {
        "run_id": directory.name,
        "target": "python",
        "recipe": cfg["name"],
        "gpu_count": cfg["execution"]["gpu_count"],
        "contract": quality["contract"],
        "correctness": quality,
        "variants": variants,
        "resolved_config": cfg.data,
        "artifact": deployment_artifact(cfg),
        "inputs_sha256": {key: file_sha(cfg["inputs"][key]) for key in ("validation", "benchmark")},
        "measurement": "End-to-end prompt-to-PIL, synchronized across GPUs; PNG writing excluded. Each image counted once. Fresh per-rank compiler caches; artifacts retain tuning-time state.",
    }
    write_json(directory / "report.json", report)
    write_json(cfg.output / "benchmark-latest.json", {"report": str(directory / "report.json")})
    print(directory / "report.json", flush=True)


def infer(args: Namespace, cfg: Config) -> None:
    artifact = deployment_artifact(cfg)
    if artifact["contract"] != contract(cfg, source_identity(args, cfg)):
        raise ValueError("Artifact contract differs from requested source/configuration")
    directory = new_run(cfg, "infer")
    ranks = launch(args, cfg, "infer", "aitune", directory / "aitune")
    write_json(directory / "report.json", {"contract": artifact["contract"], "ranks": ranks})
    print(rank_directory(cfg, directory / "aitune") / "images", flush=True)
