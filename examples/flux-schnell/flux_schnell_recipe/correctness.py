# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared image metric and isolated fresh-artifact correctness."""

from pathlib import Path
from collections import deque
import subprocess
import sys

from .data import read_inputs
from .records import artifact_metadata, contract, file_sha, new_run, read_json, source_identity, write_json


def score_images(reference, actual):
    import numpy as np
    from skimage.metrics import structural_similarity
    a, b = np.asarray(reference.convert("RGB")), np.asarray(actual.convert("RGB"))
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError(f"Image shapes differ: {a.shape} and {b.shape}")
    score = float(structural_similarity(a, b, data_range=255, channel_axis=-1))
    if not np.isfinite(score):
        raise ValueError("SSIM produced a nonfinite score")
    return score


def compare(cfg, reference, actual, records):
    from PIL import Image
    scores = []
    for item in records:
        with Image.open(Path(reference) / f"{item['id']}.png") as a, Image.open(Path(actual) / f"{item['id']}.png") as b:
            expected_size = (cfg["workload"]["width"], cfg["workload"]["height"])
            if a.size != expected_size or b.size != expected_size:
                raise ValueError("Image dimensions differ from the recipe")
            score = score_images(a, b)
        scores.append({"id": item["id"], "ssim": score, "passed": score >= cfg["validation"]["threshold"]})
    return {"metric": "rgb_ssim", "threshold": cfg["validation"]["threshold"],
            "passed": all(s["passed"] for s in scores), "scores": scores}


def worker_command(args, cfg, action, variant, run_dir):
    source = ["--model-id", args.model_id] if args.model_id else ["--checkpoint", str(Path(args.checkpoint).resolve())]
    return [sys.executable, "-m", "flux_schnell_recipe.worker", "--config", str(cfg.path),
            *source, "--action", action, "--variant", variant, "--run-dir", str(run_dir)]


def run_worker(args, cfg, action, variant, run_dir):
    command = worker_command(args, cfg, action, variant, run_dir)
    log_path = run_dir / f"{variant}.log"
    try:
        with log_path.open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as exc:
        with log_path.open(errors="replace") as log:
            tail = "".join(deque(log, maxlen=30))
        raise RuntimeError(
            f"{action} worker ({variant}) exited with status {exc.returncode}. "
            f"Full log: {log_path}\nLast log lines:\n{tail}"
        ) from exc


def run(args, cfg):
    run_dir = new_run(cfg, "correctness")
    metadata = artifact_metadata(cfg)
    expected = contract(cfg, source_identity(args, cfg))
    if metadata["contract"] != expected:
        raise ValueError("Artifact contract differs from requested source/configuration")
    for variant in ("original", "aitune"):
        run_worker(args, cfg, "correctness", variant, run_dir)
    result = compare(cfg, run_dir / "original/images", run_dir / "aitune/images", read_inputs(cfg["inputs"]["validation"]))
    result.update({"run_id": run_dir.name, "contract": expected, "artifact_sha256": metadata["sha256"],
                   "validation_inputs_sha256": file_sha(cfg["inputs"]["validation"]),
                   "original": str(run_dir / "original"), "aitune": str(run_dir / "aitune")})
    write_json(run_dir / "report.json", result)
    print(run_dir / "report.json")
    if not result["passed"]:
        raise RuntimeError("Image correctness failed; inspect the saved images and scores")
    write_json(cfg.output / "correctness-latest.json", {"report": str(run_dir / "report.json")})
    return result


def require_correctness(cfg, metadata=None):
    result = read_json(read_json(cfg.output / "correctness-latest.json")["report"])
    metadata = metadata or artifact_metadata(cfg)
    if (not result["passed"] or result["contract"] != metadata["contract"]
            or result["artifact_sha256"] != metadata["sha256"]
            or result["validation_inputs_sha256"] != file_sha(cfg["inputs"]["validation"])
            or result["threshold"] != cfg["validation"]["threshold"]):
        raise ValueError("Correctness evidence is stale; run recipe correctness with this configuration")
    return result
