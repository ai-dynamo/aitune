# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public commands; model imports occur only inside their owning process."""

import argparse
import importlib
import logging

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="FLUX.1-schnell AITune recipe")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("tune", "correctness", "benchmark", "infer", "serve", "deployment-correctness", "deployment-benchmark"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        if name in ("tune", "correctness", "benchmark", "infer", "serve"):
            source = command.add_mutually_exclusive_group(required=True)
            source.add_argument("--model-id")
            source.add_argument("--checkpoint")
        if name in ("serve", "deployment-correctness", "deployment-benchmark"):
            command.add_argument("--target", required=True, choices=("dynamo",))
            command.add_argument("--variant", required=True, choices=("original", "aitune"))
        if name.startswith("deployment-"):
            command.add_argument("--endpoint", required=True, help="Dynamo base URL, e.g. http://localhost:8000")
    args = parser.parse_args()
    cfg = load_config(args.config)
    logging.basicConfig(level=logging.INFO)
    module = {"infer": "inference", "serve": "deployment.dynamo.server",
              "deployment-correctness": "deployment.dynamo.client",
              "deployment-benchmark": "deployment.dynamo.benchmark"}.get(args.command, args.command)
    importlib.import_module(f"flux_schnell_recipe.{module}").run(args, cfg)


if __name__ == "__main__":
    main()
