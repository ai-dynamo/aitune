# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve the selected recipe's tuned FLUX pipeline with NVIDIA Dynamo."""

from __future__ import annotations

import io
import os
import threading
from logging import basicConfig
from pathlib import Path
from typing import TYPE_CHECKING, cast

from PIL import Image

import aitune.dynamo as dyn

from ..checkpoint import check_checkpoint, load_checkpoint
from ..config import GenerationInput, Recipe, argument_parser, load_recipe
from ..distributed import initialize, is_rank_zero, launch, shutdown
from ..model import generate, get_pipeline

if TYPE_CHECKING:
    from diffusers import FluxPipeline
    from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
    from dynamo.common.protocols.image_protocol import NvCreateImageRequest


class FluxDynamoBackend:
    def __init__(self, recipe: Recipe) -> None:
        self.recipe = recipe
        self._pipeline: FluxPipeline | None = None
        self._generation_lock = threading.Lock()

    def run(self) -> None:
        initialize()
        try:
            config = dyn.DynamoWorkerConfig(
                type="image",
                model_path=self.recipe.model.base_model,
                mapping=self.map_request,
            )
            dyn.dynamo_worker(self.generate, config, setup=self.setup, warmup=self.warmup)
        finally:
            shutdown()

    def setup(self) -> None:
        check_checkpoint(self.recipe)
        self._pipeline = load_checkpoint(get_pipeline(self.recipe), self.recipe)

    def warmup(self) -> None:
        sample = self.recipe.samples("inference")[0]
        self._run_pipeline(sample.prompt, sample.seed)
        pid_file = os.environ.get("AITUNE_DYNAMO_RANK_ZERO_PID_FILE")
        if pid_file and is_rank_zero():
            Path(pid_file).write_text(str(os.getpid()))

    def map_request(self, request: NvCreateImageRequest) -> dict[str, str | int]:
        workload = self.recipe.workload
        expected_size = f"{workload.width}x{workload.height}"
        if request.size and request.size != expected_size:
            raise ValueError(f"This recipe was tuned for {expected_size}")
        if request.n not in (None, 1):
            raise ValueError("This recipe generates one image per request")
        seed = request.nvext.seed if request.nvext and request.nvext.seed is not None else self.recipe.deployment.seed
        return GenerationInput(prompt=request.prompt, seed=seed).model_dump()

    def generate(self, prompt: str, seed: int) -> bytes | None:
        result = self._run_pipeline(prompt, seed)
        if not is_rank_zero():
            return None
        buffer = io.BytesIO()
        image = cast(Image.Image, result.images[0])  # Rank zero requests PIL output.
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _run_pipeline(self, prompt: str, seed: int) -> FluxPipelineOutput:
        if self._pipeline is None:
            raise RuntimeError("FLUX pipeline is not initialized")
        with self._generation_lock:
            return generate(self._pipeline, self.recipe, GenerationInput(prompt=prompt, seed=seed))


def main() -> None:
    basicConfig(level=os.environ.get("AITUNE_LOG_LEVEL", "INFO"), force=True)
    recipe = load_recipe(argument_parser(__doc__).parse_args().recipe)
    check_checkpoint(recipe)
    launch(recipe, "flux_schnell.dynamo.backend")
    FluxDynamoBackend(recipe).run()


if __name__ == "__main__":
    main()
