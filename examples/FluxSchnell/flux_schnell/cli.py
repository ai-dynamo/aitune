# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tune, validate, infer, and serve FLUX.1-schnell using recipe settings."""

import argparse

from .config import argument_parser, load_recipe
from .dynamo import serve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("tune", "Inspect and tune a FLUX pipeline on one or multiple GPUs."),
        ("validate", "Compare original and tuned images using the recipe's validation samples."),
        ("infer", "Generate images from the tuned checkpoint using the recipe's inference samples."),
    ):
        commands.add_parser(
            name,
            help=description,
            description=description,
            parents=[argument_parser(description)],
            add_help=False,
        )
    commands.add_parser(
        "serve",
        help="Keep Dynamo and the prompt UI running on one GPU, or run a smoke test.",
        description=serve.__doc__,
        parents=[serve.command_parser()],
        add_help=False,
    )
    args = parser.parse_args()
    recipe = load_recipe(args.recipe)
    if args.command == "serve":
        serve.run(recipe, smoke_test=args.smoke_test)
        return
    if args.command == "tune":
        from .tune import run
    elif args.command == "validate":
        from .validate import run
    else:
        from .infer import run
    run(recipe)


if __name__ == "__main__":
    main()
