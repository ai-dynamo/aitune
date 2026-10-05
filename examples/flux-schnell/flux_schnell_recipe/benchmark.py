# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare isolated Python variants after fresh-artifact correctness."""

from . import correctness
from .records import artifact_metadata, new_run, read_json, write_json


def run(args, cfg):
    quality = correctness.run(args, cfg)
    run_dir = new_run(cfg, "benchmark")
    variants = {}
    for variant in ("original", "aitune"):
        correctness.run_worker(args, cfg, "benchmark", variant, run_dir)
        variants[variant] = read_json(run_dir / variant / "report.json")
    write_json(run_dir / "report.json", {"run_id": run_dir.name, "target": "python", "recipe": cfg["name"],
        "correctness": quality, "variants": variants, "artifact": artifact_metadata(cfg),
        "measurement": "End-to-end prompt-to-PIL image, synchronized CUDA; PNG writing excluded; sequential batch 1"})
    print(run_dir / "report.json")
