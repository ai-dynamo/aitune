# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public commands; model imports occur only inside their owning process."""

import argparse
import importlib
import logging
import os

from .config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="FLUX.1-schnell AITune recipe")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "tune",
        "correctness",
        "benchmark",
        "infer",
        "serve",
        "web",
        "deployment-correctness",
        "deployment-benchmark",
    ):
        command = commands.add_parser(name)
        command.add_argument(
            "--config", default="recipes/single-gpu.yaml", help="Recipe YAML (default: recipes/single-gpu.yaml)"
        )
        if name in ("tune", "correctness", "benchmark", "infer", "serve"):
            source = command.add_mutually_exclusive_group()
            source.add_argument("--model-id")
            source.add_argument("--checkpoint")
        if name in ("serve", "deployment-correctness", "deployment-benchmark"):
            command.add_argument("--target", default="dynamo", choices=("dynamo",))
            command.add_argument("--variant", default="aitune", choices=("original", "aitune"))
        if name.startswith("deployment-"):
            command.add_argument("--endpoint", help="Dynamo base URL; defaults to localhost at deployment.port")
        if name == "web":
            command.add_argument("--endpoint", help="Dynamo base URL; defaults to localhost at deployment.port")
            command.add_argument("--variant", choices=("original", "aitune"), default="aitune")
            command.add_argument("--host", default="127.0.0.1", help="Bind address; use 0.0.0.0 inside Docker")
            command.add_argument("--port", type=int, default=8080, help="Web app port (default: 8080)")
    return parser, parser.parse_args()


def main():
    parser, args = parse_args()
    cfg = load_config(args.config)
    logging.basicConfig(level=logging.INFO)
    if args.command in ("tune", "correctness", "benchmark", "infer", "serve"):
        if int(os.environ.get("WORLD_SIZE", "1")) != 1 or "RANK" in os.environ:
            parser.error("Run recipe once; it launches its own workers. Do not wrap it in torchrun")
        if not args.model_id and not args.checkpoint:
            args.model_id = cfg["model"]["base_model"]
    if (
        args.command not in ("tune", "correctness", "benchmark", "infer")
        and "dynamo" not in cfg["deployment"]["targets"]
    ):
        parser.error("Enable Dynamo deployment in the config; see recipes/multi-gpu.yaml for deployment settings")
    if hasattr(args, "endpoint") and not args.endpoint:
        args.endpoint = f"http://localhost:{cfg['deployment']['port']}"
    module, function = {
        "tune": ("tune", "run"),
        "correctness": ("evaluate", "correctness"),
        "benchmark": ("evaluate", "benchmark"),
        "infer": ("evaluate", "infer"),
        "serve": ("dynamo.server", "run"),
        "web": ("web", "run"),
        "deployment-correctness": ("dynamo.client", "correctness"),
        "deployment-benchmark": ("dynamo.client", "benchmark"),
    }[args.command]
    getattr(importlib.import_module(f"flux_schnell_recipe.{module}"), function)(args, cfg)


if __name__ == "__main__":
    main()
