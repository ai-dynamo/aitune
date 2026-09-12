# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve a tuned WAN pipeline with NVIDIA Dynamo."""

import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import torch
from diffusers.utils import export_to_video

import aitune.dynamo as dyn
import aitune.torch as ait

from ..context_parallel import ContextParallelMode
from ..defaults import (
    DEFAULT_FPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_PROMPT,
    DEFAULT_WIDTH,
)
from ..distributed import distributed_output_path, is_rank_zero
from ..distributed import initialize as initialize_distributed
from ..distributed import shutdown as shutdown_distributed
from ..model import get_pipeline
from .config import load_config, validate_generation_shape

_RANK_ZERO_PID_FILE_ENV = "AITUNE_DYNAMO_RANK_ZERO_PID_FILE"


class WanDynamoBackend:
    """Serve a tuned WAN pipeline through AITune's Dynamo worker."""

    def __init__(self, config: dict) -> None:
        self._settings = config
        self.model_name = config.get("model_name", DEFAULT_MODEL_NAME)
        self._tuned_model_path = config.get("tuned_model_path", "wan2.1-t2v-1.3b.ait")
        self._context_parallel = ContextParallelMode(config.get("context_parallel", ContextParallelMode.ULYSSES))
        self._default_prompt = config.get("prompt", DEFAULT_PROMPT)
        self._default_negative_prompt = config.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
        self._default_height = config.get("height", DEFAULT_HEIGHT)
        self._default_width = config.get("width", DEFAULT_WIDTH)
        self._default_num_frames = config.get("num_frames", DEFAULT_NUM_FRAMES)
        self._default_steps = config.get("steps", DEFAULT_INFERENCE_STEPS)
        self._default_guidance_scale = config.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)
        self._default_max_sequence_length = config.get("max_sequence_length", DEFAULT_MAX_SEQUENCE_LENGTH)
        self._default_fps = config.get("fps", DEFAULT_FPS)
        self._multi_gpu = False
        self._pipeline = None
        self._generation_lock = threading.Lock()

    def run(self) -> None:
        """Initialize distributed execution and start serving requests."""
        self._multi_gpu = initialize_distributed()
        try:
            config = dyn.DynamoWorkerConfig(type="video", model_path=self.model_name, mapping=self.map_request)
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
        """Map a Dynamo video request to generation arguments."""
        width, height = (
            int(value) for value in (request.size or f"{self._default_width}x{self._default_height}").split("x")
        )
        fps = self._request_value(request, "fps", self._default_fps)
        num_frames = self._request_value(request, "num_frames", None)
        if num_frames is None:
            num_frames = request.seconds * fps + 1 if request.seconds is not None else self._default_num_frames
        if (num_frames - 1) % 4:
            raise ValueError("WAN requires the number of frames to equal 4k + 1")
        validate_generation_shape(self._settings, width=width, height=height, num_frames=num_frames)
        return {
            "prompt": request.prompt,
            "negative_prompt": self._request_value(request, "negative_prompt", self._default_negative_prompt),
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": self._request_value(request, "num_inference_steps", self._default_steps),
            "guidance_scale": self._request_value(request, "guidance_scale", self._default_guidance_scale),
            "max_sequence_length": self._default_max_sequence_length,
            "fps": fps,
            "seed": self._request_value(request, "seed", 0),
        }

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        height: int,
        width: int,
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        max_sequence_length: int,
        fps: int,
        seed: int,
    ) -> bytes | None:
        """Generate and encode one video on rank zero."""
        frames = self._run_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
            seed=seed,
        )
        if not is_rank_zero():
            return None
        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / "output.mp4"
            export_to_video(frames, output_path, fps=fps)
            return output_path.read_bytes()

    def _run_pipeline(
        self,
        prompt: str,
        negative_prompt: str,
        height: int,
        width: int,
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        max_sequence_length: int,
        seed: int,
    ):
        """Execute the pipeline under the per-process generation lock."""
        if self._pipeline is None:
            raise RuntimeError("WAN pipeline is not initialized")
        with self._generation_lock:
            return self._pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
                generator=torch.Generator("cpu").manual_seed(seed),
                output_type="np",
            ).frames[0]

    def _warmup_parameters(self) -> dict:
        """Return the lightweight generation arguments used for warmup."""
        return {
            "prompt": self._default_prompt,
            "negative_prompt": self._default_negative_prompt,
            "height": self._default_height,
            "width": self._default_width,
            "num_frames": self._default_num_frames,
            "num_inference_steps": 2,
            "guidance_scale": self._default_guidance_scale,
            "max_sequence_length": self._default_max_sequence_length,
            "seed": 0,
        }

    @staticmethod
    def _publish_rank_zero_pid() -> None:
        """Publish the endpoint-owning process ID for the launcher cleanup trap."""
        pid_file = os.environ.get(_RANK_ZERO_PID_FILE_ENV)
        if pid_file and is_rank_zero():
            Path(pid_file).write_text(str(os.getpid()))

    @staticmethod
    def _request_value(request: Any, name: str, default: Any) -> Any:
        """Read an optional Dynamo video extension with an example default."""
        value = getattr(request.nvext, name, None) if request.nvext is not None else None
        return default if value is None else value


def main() -> None:
    """Load the example configuration and run the WAN Dynamo backend."""
    WanDynamoBackend(load_config()).run()


if __name__ == "__main__":
    main()
