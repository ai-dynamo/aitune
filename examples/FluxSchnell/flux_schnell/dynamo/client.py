# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Send the recipe's inference samples through the Dynamo image endpoint."""

import base64
import json
import time
from typing import TypedDict

from openai import OpenAI

from ..config import Recipe, argument_parser, load_recipe


class DeploymentRecord(TypedDict):
    sample: str
    seed: int
    image: str
    latency_s: float


def run(recipe: Recipe) -> None:
    client = OpenAI(base_url=recipe.endpoint + "/v1", api_key="unused", timeout=600)
    samples = recipe.samples("inference")
    workload = recipe["workload"]
    output_dir = recipe.output_dir / "deployment"
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[DeploymentRecord] = []
    for index in range(recipe["deployment"]["request_count"]):
        sample = samples[index % len(samples)]
        started = time.perf_counter()
        response = client.images.generate(
            model=recipe["model"]["base_model"],
            prompt=sample["prompt"],
            n=1,
            size=f"{workload['width']}x{workload['height']}",
            response_format="b64_json",
            extra_body={"nvext": {"seed": sample["seed"]}},
        )
        image = base64.b64decode(response.data[0].b64_json, validate=True)
        if not image.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Dynamo did not return a PNG")
        path = output_dir / f"{index:03d}-{sample['id']}.png"
        path.write_bytes(image)
        records.append({
            "sample": sample["id"],
            "seed": sample["seed"],
            "image": path.name,
            "latency_s": time.perf_counter() - started,
        })
        print(f"Image saved to {path}", flush=True)
    (output_dir / "report.json").write_text(json.dumps(records, indent=2) + "\n")


def main() -> None:
    run(load_recipe(argument_parser(__doc__).parse_args().recipe))


if __name__ == "__main__":
    main()
