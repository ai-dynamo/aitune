# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tune, validate, infer, and serve FLUX.1-schnell using recipe settings."""

import argparse
from pathlib import Path

from .config import load_recipe


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("tune", "Inspect and tune a FLUX pipeline on one or multiple GPUs."),
        ("validate", "Compare original and tuned images using the recipe's validation samples."),
        ("infer", "Generate images from the tuned checkpoint using the recipe's inference samples."),
        ("serve", "Keep Dynamo and the prompt UI running on one GPU, or run a smoke test."),
        ("backend", "Run the Dynamo backend worker."),
        ("web", "Run the browser UI for an existing Dynamo deployment."),
        ("client", "Send inference samples to an existing Dynamo deployment."),
    ):
        command_parser = commands.add_parser(
            name,
            help=description,
            description=description,
        )
        command_parser.add_argument(
            "--recipe", type=Path, default=Path("recipes/single-gpu.yaml"), help="Recipe YAML path"
        )
        if name == "serve":
            command_parser.add_argument(
                "--smoke_test", action="store_true", help="Run sample requests against Dynamo, then shut it down"
            )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    recipe = load_recipe(args.recipe)
    command_args = {}
    if args.command == "serve":
        from .dynamo.serve import run

        command_args["smoke_test"] = args.smoke_test
    elif args.command == "tune":
        from .tune import run
    elif args.command == "validate":
        from .validate import run
    elif args.command == "infer":
        from .infer import run
    elif args.command == "backend":
        from .dynamo.backend import run
    elif args.command == "client":
        from .dynamo.client import run
    else:
        from .dynamo.web import run
    run(recipe, **command_args)


if __name__ == "__main__":
    main()
