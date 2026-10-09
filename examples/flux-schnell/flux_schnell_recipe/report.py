# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render an accepted two-GPU Python report into the designated README block; no GPU imports."""

import argparse
import math
import os
from pathlib import Path
from urllib.parse import quote

from .config import load_config
from .records import file_sha, read_json

START = "<!-- multi-gpu-results:start -->"
END = "<!-- multi-gpu-results:end -->"
METRICS = ("throughput_images_s", "latency_mean_ms", "peak_gpu_memory_gib",
           "gpu_utilization_pct", "compilation_s", "warmup_s")


def validate(report, cfg):
    if (report.get("schema") != "flux-multi-gpu-v1" or report.get("target") != "python"
            or report.get("gpu_count") != 2 or report.get("recipe") != cfg["name"]):
        raise ValueError("Expected a completed two-GPU Python benchmark from this recipe")
    for key in ("model", "execution", "workload", "precision"):
        if report["contract"][key] != cfg[key]:
            raise ValueError(f"Report {key} differs from the selected recipe")
    if report["resolved_config"]["benchmark"] != cfg["benchmark"]:
        raise ValueError("Report benchmark settings differ from the selected recipe")
    for phase in ("benchmark", "validation"):
        if report["inputs_sha256"][phase] != file_sha(cfg["inputs"][phase]):
            raise ValueError(f"Report {phase} inputs differ from the selected recipe")
    if not report["correctness"]["passed"] or len(report["correctness"]["ranks"]) != 2:
        raise ValueError("Both ranks must pass fresh-artifact correctness")
    qualities = list(report["correctness"]["ranks"])
    for variant in ("original", "aitune"):
        result = report["variants"][variant]
        if not result["correctness"]["passed"] or len(result["correctness"]["ranks"]) != 2:
            raise ValueError(f"{variant} failed fresh benchmark-model correctness")
        qualities.extend(result["correctness"]["ranks"])
        if len(result["gpus"]) != 2 or len({gpu["gpu_uuid"] for gpu in result["gpus"]}) != 2:
            raise ValueError("Report must contain two distinct GPU measurements")
        if len(result["ranks"]) != 2 or [r["rank"] for r in result["ranks"]] != [0, 1]:
            raise ValueError("Missing rank environment evidence")
        if result["images"] != cfg["benchmark"]["repetitions"] or result["batch_size"] != 1 or result["concurrency"] != 1:
            raise ValueError("Incomplete or different benchmark workload")
        for key in METRICS:
            value = result[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Missing/invalid {variant}.{key}; incomplete measurements cannot fill the table")
        for rank, gpu in zip(result["ranks"], result["gpus"]):
            if gpu["missing_reason"] or gpu["gpu_uuid"] != rank["environment"]["gpu"]["uuid"]:
                raise ValueError("GPU telemetry does not match rank environment")
            for key in ("peak_gpu_memory_gib", "gpu_utilization_pct"):
                if type(gpu[key]) not in (int, float) or not math.isfinite(gpu[key]) or gpu[key] < 0:
                    raise ValueError("Invalid per-device GPU measurement")
    if ([g["gpu_uuid"] for g in report["variants"]["original"]["gpus"]]
            != [g["gpu_uuid"] for g in report["variants"]["aitune"]["gpus"]]):
        raise ValueError("Original Model and AITune used different devices")
    for quality in qualities:
        if (not quality["passed"] or quality["threshold"] != cfg["validation"]["threshold"]
                or quality["soft_threshold"] != cfg["validation"].get("soft_threshold") or not quality["scores"]):
            raise ValueError("Correctness evidence uses different thresholds or has no passing scores")
        if any(not score["passed"] or not math.isfinite(score["ssim"])
               or score["ssim"] < quality["threshold"] for score in quality["scores"]):
            raise ValueError("A correctness score failed the hard threshold")


def render(report, report_path, readme_path, cfg):
    validate(report, cfg)
    link = quote(os.path.relpath(report_path, readme_path.parent), safe="/.-_")
    variants = report["variants"]
    ranks = variants["aitune"]["ranks"]
    names = ", ".join(rank["environment"]["gpu"]["name"] for rank in ranks)
    w = cfg["workload"]
    rows = [START, "", f"Python results — `{cfg['name']}`, two GPUs ({names}), BF16, Ulysses degree 2,",
            f"{w['width']}×{w['height']}, {w['num_inference_steps']} steps, text length {w['max_sequence_length']}, "
            f"batch/concurrency 1, {cfg['benchmark']['repetitions']} completed images per variant.",
            f"[Benchmark report]({link}) (run `{report['run_id']}`).", "",
            "| Variant | Images/s | Mean latency (ms) | Peak GPU memory (GiB) | GPU utilization (%) | Compilation (s) | Warmup (s) |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for name, label in (("original", "Original Model"), ("aitune", "AITune")):
        values = variants[name]
        formatted = [f"{values[key]:.{4 if index == 0 else 2}f}" for index, key in enumerate(METRICS)]
        rows.append("| " + " | ".join([label, *formatted]) + " |")
    rows += ["", "Memory is the maximum per-device sampled peak; utilization is the mean across devices.",
             "Compilation counts overlapping compiler intervals across ranks once; warmup covers the full distributed phase.",
             "Throughput counts each generated image once. Both variants passed the hard correctness gate on both ranks.", "",
             "| Variant | Rank | GPU UUID | Peak memory (GiB) | Utilization (%) |",
             "|---|---:|---|---:|---:|"]
    for name, label in (("original", "Original Model"), ("aitune", "AITune")):
        for rank, gpu in enumerate(variants[name]["gpus"]):
            rows.append(f"| {label} | {rank} | `{gpu['gpu_uuid']}` | {gpu['peak_gpu_memory_gib']:.2f} | {gpu['gpu_utilization_pct']:.2f} |")
    scores = [score["ssim"] for quality in report["correctness"]["ranks"] for score in quality["scores"]]
    soft = cfg["validation"].get("soft_threshold")
    rows += ["", f"Fresh-artifact AITune RGB SSIM: {min(scores):.4f}–{max(scores):.4f}; "
             f"hard threshold {cfg['validation']['threshold']}; soft threshold {soft}.",
             "Soft-threshold warnings remain recorded in each rank's correctness evidence.",
             f"Tuning/search: {report['tuning_search_s']:.2f} s, separate from runtime compilation.",
             "Fresh per-rank compiler caches were used; AITune retains saved tuning state and may reuse tuning-time TensorRT timing caches.",
             "Raw per-rank compiler intervals, latency records, NVML samples, backend selections, inputs, and effective environments are retained with the report.", ""]
    env = ranks[0]["environment"]
    rows += [f"Recorded rank-0 environment: driver `{env['driver']}`, CUDA `{env['cuda']}`; " +
             ", ".join(f"{name} `{env['packages'][name]}`" for name in ("aitune", "torch", "diffusers", "torch-tensorrt", "tensorrt")) + ".",
             f"Image RepoDigest: `{env['container_digest']}`." if env["container_digest"] else
             "Image RepoDigest is unavailable; retain the image ID and build provenance from the evaluation bundle.",
             "", END]
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--report", help="Explicit benchmark report.json")
    source.add_argument("--results-root", help="Multi-GPU results directory containing benchmark-latest.json")
    parser.add_argument("--readme", required=True, help="README used for relative links and optional update")
    parser.add_argument("--update-readme", action="store_true", help="Replace only the marked multi-GPU results block")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.results_root:
        root = Path(args.results_root).resolve()
        report_path = (root / read_json(root / "benchmark-latest.json")["report"]).resolve()
        if not report_path.is_relative_to(root):
            raise ValueError("Latest report must be inside --results-root")
    else:
        report_path = Path(args.report).resolve()
    readme_path = Path(args.readme).resolve()
    block = render(read_json(report_path), report_path, readme_path, cfg)
    if args.update_readme:
        text = readme_path.read_text()
        if text.count(START) != 1 or text.count(END) != 1 or text.index(START) >= text.index(END):
            raise ValueError("README must contain exactly one ordered multi-GPU results marker pair")
        updated = text[:text.index(START)] + block + text[text.index(END) + len(END):]
        readme_path.write_text(updated)
        print(f"Updated {readme_path} from {report_path}")
    else:
        print(block)


if __name__ == "__main__":
    main()
