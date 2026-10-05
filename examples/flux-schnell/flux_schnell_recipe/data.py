# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt manifests and lossless image storage."""

import json
from pathlib import Path
import re


def read_inputs(path):
    records, ids = [], set()
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or set(item) != {"id", "prompt", "seed"}:
            raise ValueError(f"{path}:{line_no}: expected exactly id, prompt, seed")
        if not isinstance(item["id"], str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", item["id"]):
            raise ValueError(f"{path}:{line_no}: invalid image ID")
        if item["id"] in ids:
            raise ValueError(f"Duplicate ID: {item['id']}")
        if not isinstance(item["prompt"], str) or not item["prompt"].strip():
            raise ValueError("prompt must be nonempty text")
        if type(item["seed"]) is not int or not 0 <= item["seed"] < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        ids.add(item["id"])
        records.append(item)
    if not records:
        raise ValueError(f"No inputs in {path}")
    return records


def save_images(pipe, cfg, records, directory):
    from .model import generate
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for item in records:
        generate(pipe, cfg, item).save(directory / f"{item['id']}.png")
