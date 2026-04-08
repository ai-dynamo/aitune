# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E5Large test client — calls the standard Dynamo OpenAI-compatible frontend."""

import argparse

from openai import OpenAI

_MODEL = "intfloat/e5-large-v2"


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--sentence", default="Hello, world!")
    args = parser.parse_args()

    client = OpenAI(
        base_url=f"http://{args.host}:{args.port}/v1",
        api_key="unused",
    )
    response = client.embeddings.create(model=_MODEL, input=args.sentence)
    print(f"Embedding dim: {len(response.data[0].embedding)}")  # noqa: T201
    print(f"First 5 values: {response.data[0].embedding[:5]}")  # noqa: T201


if __name__ == "__main__":
    main()
