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


def promote(analyzer_config: Path, deployment_repository: Path) -> tuple[str, Path, float, float]:
    """Create a deployment repository from the highest-throughput configuration."""
    config = yaml.safe_load(analyzer_config.read_text(encoding="utf-8"))
    model_name = config["profile_models"][0]
    results = Path(config["export_path"]) / "results" / "metrics-model-inference.csv"

    with results.open(newline="", encoding="utf-8") as handle:
        measurements = [row for row in csv.DictReader(handle) if row["Model"] == model_name]
    if not measurements:
        raise ValueError(f"no Model Analyzer measurements found for {model_name}")

    latency_budget_ms = config.get("latency_budget")
    if latency_budget_ms is not None:
        measurements = [row for row in measurements if float(row["p99 Latency (ms)"]) <= latency_budget_ms]
        if not measurements:
            raise ValueError(
                f"no Model Analyzer measurements for {model_name} satisfy the {latency_budget_ms} ms p99 latency budget"
            )

    best = max(measurements, key=lambda row: float(row["Throughput (infer/sec)"]))
    throughput = float(best["Throughput (infer/sec)"])
    p99_latency_ms = float(best["p99 Latency (ms)"])
    variant_name = best["Model Config Path"]
    variant_config = Path(config["output_model_repository_path"]) / variant_name / "config.pbtxt"

    triton_config = model_config_pb2.ModelConfig()
    text_format.Parse(variant_config.read_text(encoding="utf-8"), triton_config)
    triton_config.name = model_name

    deployed_model = deployment_repository / model_name
    shutil.copytree(Path(config["model_repository"]) / model_name, deployed_model)
    deployed_config = deployed_model / "config.pbtxt"
    deployed_config.write_text(text_format.MessageToString(triton_config), encoding="utf-8")
    return variant_name, deployed_config, throughput, p99_latency_ms


def main() -> None:
    """Promote the highest-throughput Model Analyzer configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyzer-config", type=Path, required=True)
    parser.add_argument("--deployment-model-repository", type=Path, required=True)
    args = parser.parse_args()
    variant, deployed_config, throughput, p99_latency_ms = promote(
        args.analyzer_config, args.deployment_model_repository
    )
    print(f"Promoted Model Analyzer configuration: {variant}", flush=True)
    print(f"Model Analyzer performance: {throughput:.2f} infer/sec, p99 latency {p99_latency_ms:.3f} ms", flush=True)
    print(
        f"Selected Triton configuration ({deployed_config}):\n{deployed_config.read_text(encoding='utf-8')}", flush=True
    )
    print(f"Deployment model repository: {args.deployment_model_repository.resolve()}", flush=True)


if __name__ == "__main__":
    main()
