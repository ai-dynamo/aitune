# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare isolated Python variants after fresh-artifact correctness."""

import logging

from . import correctness
from .records import artifact_metadata, new_run, read_json, write_json

logger = logging.getLogger(__name__)


def run(args, cfg):
    logger.info("Benchmark: running fresh correctness checks before measuring both variants")
    quality = correctness.run(args, cfg)
    run_dir = new_run(cfg, "benchmark")
    logger.info("Benchmark results: %s", run_dir)
    variants = {}
    for variant in ("original", "aitune"):
        correctness.run_worker(args, cfg, "benchmark", variant, run_dir)
        variants[variant] = read_json(run_dir / variant / "report.json")
    write_json(run_dir / "report.json", {"run_id": run_dir.name, "target": "python", "recipe": cfg["name"],
        "correctness": quality, "variants": variants, "artifact": artifact_metadata(cfg),
        "measurement": "End-to-end prompt-to-PIL image, synchronized CUDA; PNG writing excluded; sequential batch 1"})
    logger.info("Benchmark complete; report: %s", run_dir / "report.json")
    print(run_dir / "report.json")
