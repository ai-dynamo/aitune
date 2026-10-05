# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate user images from a freshly loaded artifact."""

from .correctness import run_worker
from .records import new_run


def run(args, cfg):
    directory = new_run(cfg, "infer")
    run_worker(args, cfg, "infer", "aitune", directory)
    print(directory / "aitune/images")
