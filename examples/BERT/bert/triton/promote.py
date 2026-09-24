# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Promote the highest-throughput BERT Model Analyzer configuration."""

import argparse
import csv
import shutil
from pathlib import Path

import yaml
from google.protobuf import text_format
from tritonclient.grpc import model_config_pb2


def promote(analyzer_config: Path, deployment_repository: Path) -> tuple[str, Path, dict[str, str], int]:
    """Create a deployment repository from the highest-throughput configuration."""
    config = yaml.safe_load(analyzer_config.read_text(encoding="utf-8"))
    model_name = next(iter(config["profile_models"]))
    results = Path(config["export_path"]) / "results" / "metrics-model-inference.csv"

    with results.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "Model",
            "Model Config Path",
            "Throughput (infer/sec)",
            "p90 Latency (ms)",
            "p95 Latency (ms)",
            "p99 Latency (ms)",
            "Batch",
            "Concurrency",
            "Instance Group",
        }
        missing_columns = required_columns.difference(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                f"Model Analyzer results in {results} lack {', '.join(sorted(missing_columns))}; regenerate the results"
            )
        measurements = [row for row in reader if row["Model"] == model_name]
    if not measurements:
        raise ValueError(f"no Model Analyzer measurements found for {model_name}")

    if config.get("latency_budget") is not None:
        raise ValueError("Regenerate the Model Analyzer config to use an explicit percentile latency budget")
    latency_percentile = int(config.get("perf_analyzer_flags", {}).get("percentile", 95))
    latency_budget_ms = config.get("constraints", {}).get(f"perf_latency_p{latency_percentile}", {}).get("max")
    if latency_budget_ms is not None:
        measurements = [
            row for row in measurements if float(row[f"p{latency_percentile} Latency (ms)"]) <= latency_budget_ms
        ]
        if not measurements:
            raise ValueError(
                f"no Model Analyzer measurements for {model_name} satisfy the "
                f"{latency_budget_ms} ms p{latency_percentile} latency budget"
            )

    best = max(measurements, key=lambda row: float(row["Throughput (infer/sec)"]))
    variant_name = best["Model Config Path"]
    variant_config = Path(config["output_model_repository_path"]) / variant_name / "config.pbtxt"

    triton_config = model_config_pb2.ModelConfig()
    text_format.Parse(variant_config.read_text(encoding="utf-8"), triton_config)
    triton_config.name = model_name

    deployed_model = deployment_repository / model_name
    shutil.copytree(Path(config["model_repository"]) / model_name, deployed_model)
    deployed_config = deployed_model / "config.pbtxt"
    deployed_config.write_text(text_format.MessageToString(triton_config), encoding="utf-8")
    return variant_name, deployed_config, best, latency_percentile


def main() -> None:
    """Select and display the fastest analyzed Triton configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyzer-config", type=Path, required=True)
    parser.add_argument("--deployment-model-repository", type=Path, required=True)
    args = parser.parse_args()
    variant, deployed_config, measurement, latency_percentile = promote(
        args.analyzer_config, args.deployment_model_repository
    )
    print(f"Promoted Model Analyzer configuration: {variant}", flush=True)
    print(f"Model Analyzer stabilization percentile: p{latency_percentile}", flush=True)
    print(
        "Model Analyzer workload: "
        f"request batch {measurement['Batch']}, concurrency {measurement['Concurrency']}, "
        f"instances {measurement['Instance Group']}",
        flush=True,
    )
    print(
        f"Model Analyzer performance: {float(measurement['Throughput (infer/sec)']):.2f} infer/sec; "
        f"p90 latency {float(measurement['p90 Latency (ms)']):.3f} ms, "
        f"p95 latency {float(measurement['p95 Latency (ms)']):.3f} ms, "
        f"p99 latency {float(measurement['p99 Latency (ms)']):.3f} ms (same workload)",
        flush=True,
    )
    print(
        f"Selected Triton configuration ({deployed_config}):\n{deployed_config.read_text(encoding='utf-8')}", flush=True
    )
    print(f"Deployment model repository: {args.deployment_model_repository.resolve()}", flush=True)


if __name__ == "__main__":
    main()
