# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Promote the highest-throughput Model Analyzer configuration."""

import argparse
import csv
import shutil
from pathlib import Path

import yaml
from google.protobuf import text_format
from tritonclient.grpc import model_config_pb2


def promote(analyzer_config: Path, deployment_repository: Path) -> str:
    """Create a deployment repository from the highest-throughput configuration."""
    config = yaml.safe_load(analyzer_config.read_text(encoding="utf-8"))
    model_name = config["profile_models"][0]
    results = Path(config["export_path"]) / "results" / "metrics-model-inference.csv"

    with results.open(newline="", encoding="utf-8") as handle:
        measurements = [row for row in csv.DictReader(handle) if row["Model"] == model_name]
    if not measurements:
        raise ValueError(f"no Model Analyzer measurements found for {model_name}")

    best = max(measurements, key=lambda row: float(row["Throughput (infer/sec)"]))
    variant_name = best["Model Config Path"]
    variant_config = Path(config["output_model_repository_path"]) / variant_name / "config.pbtxt"

    triton_config = model_config_pb2.ModelConfig()
    text_format.Parse(variant_config.read_text(encoding="utf-8"), triton_config)
    triton_config.name = model_name

    deployed_model = deployment_repository / model_name
    shutil.copytree(Path(config["model_repository"]) / model_name, deployed_model)
    (deployed_model / "config.pbtxt").write_text(text_format.MessageToString(triton_config), encoding="utf-8")
    return variant_name


def main() -> None:
    """Promote the highest-throughput Model Analyzer configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyzer-config", type=Path, required=True)
    parser.add_argument("--deployment-model-repository", type=Path, required=True)
    args = parser.parse_args()
    variant = promote(args.analyzer_config, args.deployment_model_repository)
    print(f"Promoted Model Analyzer configuration: {variant}", flush=True)
    print(f"Deployment model repository: {args.deployment_model_repository.resolve()}", flush=True)


if __name__ == "__main__":
    main()
