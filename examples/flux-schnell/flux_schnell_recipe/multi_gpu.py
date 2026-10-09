# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A single controller launches fresh context-parallel process groups and joins evidence."""

from collections import deque
import os
from pathlib import Path
import signal
import subprocess
import sys

from .compilation import interval_seconds
from .distributed import rank_config
from .records import artifact_metadata, contract, file_sha, new_run, read_json, source_identity, write_json


def launch(args, cfg, action, variant, directory, reference=None):
    directory.mkdir(parents=True, exist_ok=False)
    source = ["--model-id", args.model_id] if args.model_id else ["--checkpoint", str(Path(args.checkpoint).resolve())]
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               "--nproc-per-node=2", "--max-restarts=0", "--module", "flux_schnell_recipe.distributed_worker",
               "--config", str(cfg.path), *source, "--action", action, "--variant", variant,
               "--run-dir", str(directory)]
    if reference is not None:
        command += ["--reference", str(reference)]
    write_json(directory / "command.json", command)
    log_path = directory / "torchrun.log"
    print(f"{action} / {variant}: {log_path}", flush=True)
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
    if code:
        with log_path.open(errors="replace") as log:
            tail = "".join(deque(log, maxlen=30))
        raise RuntimeError(f"Two-GPU {action}/{variant} exited {code}. Full log: {log_path}\n{tail}")
    return [read_json(directory / f"rank-{rank}" / "report.json") for rank in range(2)]


def require_manifest(args, cfg):
    manifest = read_json(cfg.artifact.parent / "manifest.json")
    expected = contract(cfg, source_identity(args, cfg))
    if manifest["contract"] != expected or manifest["gpu_count"] != 2:
        raise ValueError("Multi-GPU artifact manifest differs from requested source/configuration")
    for rank in range(2):
        metadata = artifact_metadata(rank_config(cfg, rank))
        if metadata["contract"] != expected or metadata["sha256"] != manifest["artifacts"][rank]["sha256"]:
            raise ValueError("Rank artifact differs from the completed tuning manifest")
    return manifest


def correctness(args, cfg, manifest):
    from .correctness import compare
    from .data import read_inputs
    directory = new_run(cfg, "correctness")
    variants = {variant: launch(args, cfg, "correctness", variant, directory / variant)
                for variant in ("original", "aitune")}
    inputs = read_inputs(cfg["inputs"]["validation"])
    scores = [compare(cfg, directory / "original" / f"rank-{rank}" / "images",
                      directory / "aitune" / f"rank-{rank}" / "images", inputs) for rank in range(2)]
    result = {"run_id": directory.name, "contract": manifest["contract"], "gpu_count": 2,
              "artifact_sha256": [item["sha256"] for item in manifest["artifacts"]],
              "validation_inputs_sha256": file_sha(cfg["inputs"]["validation"]),
              "passed": all(score["passed"] for score in scores), "ranks": scores, "variants": variants,
              "original": str(directory / "original"), "aitune": str(directory / "aitune")}
    write_json(directory / "report.json", result)
    print(directory / "report.json", flush=True)
    if not result["passed"]:
        raise RuntimeError("Multi-GPU image correctness failed; inspect each rank's images and scores")
    write_json(cfg.output / "correctness-latest.json", {"report": str((directory / "report.json").relative_to(cfg.output))})
    return directory, result


def phase_duration(ranks, phase):
    return max(r[f"{phase}_end_monotonic_s"] for r in ranks) - min(r[f"{phase}_start_monotonic_s"] for r in ranks)


def aggregate(ranks, directory, cfg):
    if [rank["rank"] for rank in ranks] != [0, 1]:
        raise ValueError("Expected exactly ranks 0 and 1")
    if ranks[0]["contract"] != ranks[1]["contract"] or ranks[0]["workload_inputs"] != ranks[1]["workload_inputs"]:
        raise ValueError("Rank workload contracts differ")
    timings = [read_json(directory / f"rank-{rank}" / "timings.json") for rank in range(2)]
    count = cfg["benchmark"]["repetitions"]
    if any(len(rows) != count for rows in timings):
        raise ValueError("Incomplete distributed measurement")
    joined = []
    for left, right in zip(*timings):
        if left["id"] != right["id"]:
            raise ValueError("Ranks processed different inputs")
        elapsed = max(left["end_monotonic_s"], right["end_monotonic_s"]) - min(left["start_monotonic_s"], right["start_monotonic_s"])
        if elapsed <= 0:
            raise ValueError("Invalid distributed latency interval")
        joined.append({"id": left["id"], "elapsed_s": elapsed})
    duration = phase_duration(ranks, "measurement")
    gpus = [r["gpu"] for r in ranks]
    if len({gpu["gpu_uuid"] for gpu in gpus}) != 2 or any(gpu["missing_reason"] for gpu in gpus):
        raise ValueError("Require complete telemetry for two distinct GPUs")
    compiler_records = [record for rank in range(2)
                        for record in read_json(directory / f"rank-{rank}" / "compilation.json")["records"]]
    write_json(directory / "timings.json", joined)
    result = {"variant": ranks[0]["variant"], "images": count, "batch_size": 1, "concurrency": 1,
              "throughput_images_s": count / duration,
              "latency_mean_ms": sum(row["elapsed_s"] for row in joined) / count * 1000,
              "peak_gpu_memory_gib": max(gpu["peak_gpu_memory_gib"] for gpu in gpus),
              "gpu_utilization_pct": sum(gpu["gpu_utilization_pct"] for gpu in gpus) / 2,
              "compilation_s": interval_seconds(compiler_records), "warmup_s": phase_duration(ranks, "warmup"),
              "first_use_s": phase_duration(ranks, "first_use"), "measurement_duration_s": duration,
              "gpus": gpus, "ranks": ranks,
              "correctness": {"passed": all(r["correctness"]["passed"] for r in ranks),
                              "ranks": [r["correctness"] for r in ranks]},
              "cache_state": "Fresh per-rank Torch/Inductor/Triton/CUDA caches; saved AITune artifacts retain tuning state and may reuse their tuning-time TensorRT timing caches"}
    write_json(directory / "report.json", result)
    return result


def run(args, cfg):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or "RANK" in os.environ:
        raise ValueError("Run recipe once; it launches torchrun itself. Do not wrap the public CLI in torchrun")
    if args.command == "tune":
        cfg.artifact.parent.mkdir(parents=True, exist_ok=False)
        directory = new_run(cfg, "tune")
        ranks = launch(args, cfg, "tune", "original", directory / "workers")
        expected = contract(cfg, source_identity(args, cfg))
        if any(rank["contract"] != expected for rank in ranks):
            raise ValueError("Rank tuning contracts differ")
        manifest = {"contract": expected, "gpu_count": 2, "tune_run": str(directory),
                    "tuning_search_s": phase_duration(ranks, "tuning"), "ranks": ranks,
                    "artifacts": [{"rank": rank, "file": rank_config(cfg, rank).artifact.name,
                                   "sha256": ranks[rank]["sha256"]} for rank in range(2)]}
        write_json(cfg.artifact.parent / "manifest.json", manifest)
        write_json(directory / "report.json", manifest)
        print(directory / "report.json", flush=True)
        return
    manifest = require_manifest(args, cfg)
    if args.command == "correctness":
        correctness(args, cfg, manifest)
    elif args.command == "infer":
        directory = new_run(cfg, "infer")
        ranks = launch(args, cfg, "infer", "aitune", directory / "aitune")
        write_json(directory / "report.json", {"contract": manifest["contract"], "gpu_count": 2, "ranks": ranks})
        print(directory / "aitune" / "rank-0" / "images", flush=True)
    elif args.command == "benchmark":
        reference_dir, evidence = correctness(args, cfg, manifest)
        directory = new_run(cfg, "benchmark")
        variants = {}
        for variant in ("original", "aitune"):
            ranks = launch(args, cfg, "benchmark", variant, directory / variant, reference_dir / "original")
            variants[variant] = aggregate(ranks, directory / variant, cfg)
        report = {"schema": "flux-multi-gpu-v1", "run_id": directory.name, "recipe": cfg["name"],
                  "target": "python", "gpu_count": 2, "contract": manifest["contract"],
                  "resolved_config": cfg.data, "artifact_manifest": manifest,
                  "correctness": evidence, "variants": variants,
                  "inputs_sha256": {key: file_sha(cfg["inputs"][key]) for key in ("validation", "benchmark")},
                  "measurement_method": "Single-node shared clocks; both ranks cooperate on each image. Latency spans earliest rank start to latest rank completion including synchronization. Throughput counts each image once. Memory is the maximum device peak; utilization is the mean of time-weighted NVML device samples. Compilation is the union of compiler intervals across ranks. Warmup spans the full distributed phase.",
                  "tuning_search_s": manifest["tuning_search_s"]}
        write_json(directory / "report.json", report)
        write_json(cfg.output / "benchmark-latest.json", {"report": str((directory / "report.json").relative_to(cfg.output))})
        print(directory / "report.json", flush=True)
