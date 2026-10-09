# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared image metric and isolated fresh-artifact correctness."""

from pathlib import Path
from collections import deque
import logging
import subprocess
import sys
import time

from .data import read_inputs
from .records import artifact_metadata, contract, file_sha, new_run, read_json, source_identity, write_json

logger = logging.getLogger(__name__)


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
    threshold = cfg["validation"]["threshold"]
    soft_threshold = cfg["validation"].get("soft_threshold")
    scores = []
    for item in records:
        with Image.open(Path(reference) / f"{item['id']}.png") as a, Image.open(Path(actual) / f"{item['id']}.png") as b:
            expected_size = (cfg["workload"]["width"], cfg["workload"]["height"])
            if a.size != expected_size or b.size != expected_size:
                raise ValueError("Image dimensions differ from the recipe")
            score = score_images(a, b)
        passed = score >= threshold
        soft_passed = score >= soft_threshold if soft_threshold is not None else None
        if passed and soft_passed is False:
            logger.warning("Image %s: SSIM %.6f is below soft threshold %.6f (hard threshold %.6f passed)",
                           item["id"], score, soft_threshold, threshold)
        scores.append({"id": item["id"], "ssim": score, "passed": passed, "soft_passed": soft_passed})
    return {"metric": "rgb_ssim", "threshold": threshold, "soft_threshold": soft_threshold,
            "passed": all(s["passed"] for s in scores),
            "soft_passed": all(s["soft_passed"] for s in scores) if soft_threshold is not None else None,
            "scores": scores}


def worker_command(args, cfg, action, variant, run_dir):
    source = ["--model-id", args.model_id] if args.model_id else ["--checkpoint", str(Path(args.checkpoint).resolve())]
    return [sys.executable, "-u", "-m", "flux_schnell_recipe.worker", "--config", str(cfg.path),
            *source, "--action", action, "--variant", variant, "--run-dir", str(run_dir)]


def run_worker(args, cfg, action, variant, run_dir):
    command = worker_command(args, cfg, action, variant, run_dir)
    log_path = run_dir / f"{variant}.log"
    started = time.monotonic()
    heartbeat = started
    logger.info("Starting %s (%s); full log: %s", action, variant, log_path)
    try:
        with log_path.open("w") as log, log_path.open(errors="replace") as output:
            with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT) as process:
                try:
                    while True:
                        try:
                            process.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            pass
                        text = output.read()
                        if text:
                            sys.stderr.write(text)
                            sys.stderr.flush()
                        if process.returncode is not None:
                            break
                        now = time.monotonic()
                        if now - heartbeat >= 30:
                            logger.info("%s (%s) still running; elapsed %.0fs; log: %s",
                                        action, variant, now - started, log_path)
                            heartbeat = now
                    if process.returncode:
                        raise subprocess.CalledProcessError(process.returncode, command)
                except BaseException:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    raise
    except subprocess.CalledProcessError as exc:
        with log_path.open(errors="replace") as log:
            tail = "".join(deque(log, maxlen=30))
        raise RuntimeError(
            f"{action} worker ({variant}) exited with status {exc.returncode}. "
            f"Full log: {log_path}\nLast log lines:\n{tail}"
        ) from exc
    logger.info("Finished %s (%s) in %.1fs", action, variant, time.monotonic() - started)


def run(args, cfg):
    run_dir = new_run(cfg, "correctness")
    logger.info("Correctness: checking artifact and source; results: %s", run_dir)
    metadata = artifact_metadata(cfg)
    expected = contract(cfg, source_identity(args, cfg))
    if metadata["contract"] != expected:
        raise ValueError("Artifact contract differs from requested source/configuration")
    for variant in ("original", "aitune"):
        run_worker(args, cfg, "correctness", variant, run_dir)
    logger.info("Comparing Original Model and AITune images using SSIM")
    result = compare(cfg, run_dir / "original/images", run_dir / "aitune/images", read_inputs(cfg["inputs"]["validation"]))
    result.update({"run_id": run_dir.name, "contract": expected, "artifact_sha256": metadata["sha256"],
                   "validation_inputs_sha256": file_sha(cfg["inputs"]["validation"]),
                   "original": str(run_dir / "original"), "aitune": str(run_dir / "aitune")})
    write_json(run_dir / "report.json", result)
    logger.info("Correctness %s; report: %s", "passed" if result["passed"] else "failed", run_dir / "report.json")
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
