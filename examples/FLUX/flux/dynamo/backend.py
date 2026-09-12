# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve a tuned FLUX pipeline with NVIDIA Dynamo."""

import io
import os
import threading
from pathlib import Path
from typing import Any

import torch
import yaml

import aitune.dynamo as dyn
import aitune.torch as ait

from ..context_parallel import ContextParallelMode
from ..defaults import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_PROMPT,
)
from ..distributed import distributed_output_path, is_rank_zero
from ..distributed import initialize as initialize_distributed
from ..distributed import shutdown as shutdown_distributed
from ..model import MODEL_NAME, get_pipeline

_RANK_ZERO_PID_FILE_ENV = "AITUNE_DYNAMO_RANK_ZERO_PID_FILE"


class FluxDynamoBackend:
    """Serve a tuned FLUX pipeline through AITune's Dynamo worker."""

    def __init__(self, config: dict) -> None:
        backend_config = config.get("Backend", {})
        tuned_model_path = backend_config.get("tuned_model_path")
        if not tuned_model_path:
            raise ValueError("Backend.tuned_model_path must be configured")
        sizes = backend_config.get("sizes", [DEFAULT_IMAGE_SIZE])
        if not sizes:
            raise ValueError("Backend.sizes must contain at least one entry")

        self.model_name = backend_config.get("model_name", MODEL_NAME)
        self._tuned_model_path = tuned_model_path
        self._context_parallel = ContextParallelMode(
            backend_config.get("context_parallel", ContextParallelMode.ULYSSES)
        )
        self._default_width, self._default_height = sizes[0]
        self._default_prompt = backend_config.get("prompt", DEFAULT_PROMPT)
        self._inference_steps = backend_config.get("steps", DEFAULT_INFERENCE_STEPS)
        self._guidance_scale = backend_config.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)
        self._max_sequence_length = backend_config.get("max_sequence_length", DEFAULT_MAX_SEQUENCE_LENGTH)
        self._multi_gpu = False
        self._pipeline = None
        self._generation_lock = threading.Lock()

    def run(self) -> None:
        """Initialize distributed execution and start serving requests."""
        self._multi_gpu = initialize_distributed()
        try:
            config = dyn.DynamoWorkerConfig(type="image", model_path=self.model_name, mapping=self.map_request)
            dyn.dynamo_worker(self.generate, config, setup=self.setup, warmup=self.warmup)
        finally:
            shutdown_distributed()

    def setup(self) -> None:
        """Load the pipeline and its tuned checkpoint on every rank."""
        pipeline = get_pipeline(
            self.model_name,
            multi_gpu=self._multi_gpu,
            context_parallel=self._context_parallel,
        )
        checkpoint_path = distributed_output_path(self._tuned_model_path) if self._multi_gpu else self._tuned_model_path
        self._pipeline = ait.load(pipeline, checkpoint_path)

    def warmup(self) -> None:
        """Warm the shared pipeline execution path before serving."""
        self._run_pipeline(**self._warmup_parameters())
        self._publish_rank_zero_pid()

    def map_request(self, request: Any) -> dict:
        """Map a Dynamo image request to generation arguments."""
        width, height = (
            int(value) for value in (request.size or f"{self._default_width}x{self._default_height}").split("x")
        )
        return {
            "prompt": request.prompt,
            "height": height,
            "width": width,
            "num_inference_steps": self._inference_steps,
            "guidance_scale": self._guidance_scale,
            "max_sequence_length": self._max_sequence_length,
        }

    def generate(
        self,
        prompt: str,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        max_sequence_length: int,
    ) -> bytes | None:
        """Generate and encode one image on rank zero."""
        result = self._run_pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
        )
        if not is_rank_zero():
            return None
        buffer = io.BytesIO()
        result.images[0].save(buffer, format="PNG")
        return buffer.getvalue()

    def _run_pipeline(
        self,
        prompt: str,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        max_sequence_length: int,
    ):
        """Execute the pipeline under the per-process generation lock."""
        if self._pipeline is None:
            raise RuntimeError("FLUX pipeline is not initialized")
        with self._generation_lock:
            return self._pipeline(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
                generator=torch.Generator("cpu").manual_seed(0),
                output_type="pil" if is_rank_zero() else "pt",
                return_dict=True,
            )

    def _warmup_parameters(self) -> dict:
        """Return the lightweight generation arguments used for warmup."""
        return {
            "prompt": self._default_prompt,
            "height": self._default_height,
            "width": self._default_width,
            "num_inference_steps": 2,
            "guidance_scale": self._guidance_scale,
            "max_sequence_length": self._max_sequence_length,
        }

    @staticmethod
    def _publish_rank_zero_pid() -> None:
        """Publish the endpoint-owning process ID for the launcher cleanup trap."""
        pid_file = os.environ.get(_RANK_ZERO_PID_FILE_ENV)
        if pid_file and is_rank_zero():
            Path(pid_file).write_text(str(os.getpid()))


def main() -> None:
    """Load the example configuration and run the FLUX Dynamo backend."""
    FluxDynamoBackend(_get_config()).run()


def _get_config() -> dict:
    """Load the example configuration."""
    with Path(os.environ.get("AITUNE_EXAMPLE_CONFIG_PATH", "config.yaml")).open() as config_file:
        return yaml.safe_load(config_file)


if __name__ == "__main__":
    main()
