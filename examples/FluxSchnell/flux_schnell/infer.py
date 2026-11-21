# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate images from the tuned checkpoint using the recipe's inference samples."""

import os
from logging import basicConfig, getLogger

from .checkpoint import check_checkpoint, load_checkpoint
from .config import Recipe
from .distributed import initialize, is_rank_zero, shutdown
from .model import generate_samples, get_pipeline

logger = getLogger(__name__)


def do_inference(recipe: Recipe) -> None:
    check_checkpoint(recipe)
    pipeline = load_checkpoint(get_pipeline(recipe), recipe)
    generate_samples(pipeline, recipe, "inference", "tuned")
    if is_rank_zero():
        logger.info("Saved inference images to %s", recipe.output_dir / "inference" / "tuned")


def run(recipe: Recipe) -> None:
    basicConfig(level=os.environ.get("AITUNE_LOG_LEVEL", "INFO"), force=True)
    initialize(recipe)
    try:
        do_inference(recipe)
    finally:
        shutdown()
