# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FLUX test client — calls the standard Dynamo OpenAI-compatible frontend."""

from __future__ import annotations

import argparse
import base64
import pathlib

from openai import OpenAI

from ..model import MODEL_NAME


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="A futuristic cityscape at night")
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--output", default="output.png")
    args = parser.parse_args()

    client = OpenAI(
        base_url=f"http://{args.host}:{args.port}/v1",
        api_key="unused",
    )
    response = client.images.generate(
        model=MODEL_NAME,
        prompt=args.prompt,
        size=args.size,
        response_format="b64_json",
    )
    image_bytes = base64.b64decode(response.data[0].b64_json)
    pathlib.Path(args.output).write_bytes(image_bytes)
    print(f"Image saved to {args.output}")


if __name__ == "__main__":
    main()
