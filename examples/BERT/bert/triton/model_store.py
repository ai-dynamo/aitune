# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish a tuned BERT checkpoint to a Triton model repository."""

import argparse
import json
from pathlib import Path
from typing import cast

import torch
import yaml

import aitune.triton
from aitune.torch import Module, load
from bert.tune import BATCH_SIZES, SEQUENCE_LENGTHS


def _write_mixed_workload(search_config: Path) -> None:
    """Profile all token lengths in one Model Analyzer search."""
    input_data_path = search_config.parent / "input-data.json"
    requests = [
        {"input_ids": {"content": [101, *([1000] * (length - 2)), 102], "shape": [length]}}
        for length in SEQUENCE_LENGTHS
    ]
    input_data_path.write_text(json.dumps({"data": requests}) + "\n", encoding="utf-8")

    config = yaml.safe_load(search_config.read_text(encoding="utf-8"))
    # Every request has its own shape, so Perf Analyzer does not need a global --shape flag.
    config["perf_analyzer_flags"] = {"input-data": [str(input_data_path.resolve())]}
    if isinstance(config["profile_models"], dict):
        model_name = next(iter(config["profile_models"]))
        model_profile = config["profile_models"][model_name]
        model_profile["parameters"]["batch_sizes"] = BATCH_SIZES
        model_profile["model_config_parameters"]["max_batch_size"] = [max(BATCH_SIZES)]
    else:
        config["run_config_search_min_model_batch_size"] = max(BATCH_SIZES)
        config["run_config_search_max_model_batch_size"] = max(BATCH_SIZES)
    search_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> None:
    """Load the checkpoint and generate the Triton model and Analyzer config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Directory used by bert-tune")
    parser.add_argument("--model-repository", type=Path, help="Triton model repository")
    args = parser.parse_args()

    output_dir = args.output_dir or Path(__file__).resolve().parents[2] / "artifacts"
    model_repository = args.model_repository or output_dir / "model_repository"
    checkpoint = output_dir / "bert.ait"
    if not checkpoint.is_file():
        parser.error(f"Missing {checkpoint}; run bert-tune first")

    # These AOT backends restore their compiled artifacts without the original source model.
    tuned_model = cast(Module, load(torch.nn.Identity(), checkpoint))
    try:
        artifact = tuned_model.artifact()
        model_path = aitune.triton.publish(
            artifact,
            path=model_repository,
            model_name="bert",
            max_batch_size=max(BATCH_SIZES),
        )
        # The bounded search preserves all TensorRT profile indices in Model Analyzer variants.
        search_configs = aitune.triton.generate_model_analyzer_configs(
            artifact,
            model_path=model_path,
            path=model_repository.resolve().parent / f"{model_repository.name}-model-analyzer" / "bert" / "search",
        )
        _write_mixed_workload(search_configs / "fast.yaml")
        print(f"Triton model: {model_path}", flush=True)
        print(f"Model Analyzer search configuration: {search_configs / 'fast.yaml'}", flush=True)
    finally:
        tuned_model.deactivate()


if __name__ == "__main__":
    main()
