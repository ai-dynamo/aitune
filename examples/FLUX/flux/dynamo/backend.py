# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FLUX image backend — serves via dyn.dynamo_worker."""

import io
import os
import signal
import threading
from logging import getLogger
from pathlib import Path
from typing import TypedDict

import torch
import torch.distributed as dist
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
from ..model import MODEL_NAME, get_pipeline

_STOP_COMMAND = "stop"
_GENERATE_COMMAND = "generate"
_RANK_ZERO_PID_FILE_ENV = "AITUNE_DYNAMO_RANK_ZERO_PID_FILE"

logger = getLogger(__name__)


class _DynamoCommand(TypedDict):
    """Command broadcast from the Dynamo worker to follower ranks."""

    command: str
    kwargs: dict | None


def _get_config() -> dict:
    with Path(os.environ.get("AITUNE_EXAMPLE_CONFIG_PATH", "config.yaml")).open() as f:
        return yaml.safe_load(f)


def _initialize_distributed() -> bool:
    """Initialize the process group when launched with multiple Dynamo ranks."""
    if int(os.environ.get("WORLD_SIZE", "1")) == 1:
        return False
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return True


def _run_generation(pipeline, kwargs: dict):
    """Run one generation on every rank and propagate rank-local failures."""
    result = None
    error = None
    try:
        result = pipeline(
            **kwargs,
            generator=torch.Generator("cpu").manual_seed(0),
            output_type="pil" if is_rank_zero() else "pt",
            return_dict=True,
        )
    except Exception as exc:  # noqa: BLE001
        error = f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}" if dist.is_initialized() else str(exc)

    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
        failures = [failure for failure in errors if failure is not None]
        if failures:
            raise RuntimeError("Distributed FLUX generation failed: " + "; ".join(failures))
    elif error is not None:
        raise RuntimeError(f"FLUX generation failed: {error}")
    return result


def _broadcast(command: str, kwargs: dict | None = None) -> None:
    """Broadcast a rank-zero Dynamo command to the inference ranks."""
    payload: list[_DynamoCommand | None] = [{"command": command, "kwargs": kwargs}]
    dist.broadcast_object_list(payload, src=0)


def _publish_rank_zero_pid() -> None:
    """Publish the worker PID that can initiate a coordinated shutdown."""
    pid_file = os.environ.get(_RANK_ZERO_PID_FILE_ENV)
    if pid_file and is_rank_zero():
        Path(pid_file).write_text(str(os.getpid()))


def _run_follower(pipeline) -> None:
    """Execute commands received from the rank-zero Dynamo worker."""
    # torchrun forwards termination signals to every rank. Rank zero translates
    # them into a stop command after the Dynamo runtime shuts down.
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: None)
    while True:
        payload: list[_DynamoCommand | None] = [None]
        dist.broadcast_object_list(payload, src=0)
        message = payload[0]
        if message is None:
            raise RuntimeError("Received an empty distributed Dynamo command")
        if message["command"] == _STOP_COMMAND:
            return
        if message["command"] != _GENERATE_COMMAND:
            raise ValueError(f"Unknown distributed Dynamo command: {message['command']}")
        kwargs = message["kwargs"]
        if kwargs is None:
            raise ValueError("Distributed Dynamo generate command is missing kwargs")
        try:
            _run_generation(pipeline, kwargs)
        except RuntimeError:
            logger.exception("Distributed FLUX request failed")


def main() -> None:
    multi_gpu = _initialize_distributed()
    cfg = _get_config()
    backend_cfg = cfg.get("Backend", {})
    model_name = backend_cfg.get("model_name", MODEL_NAME)
    tuned_model_path = backend_cfg.get("tuned_model_path")
    context_parallel = ContextParallelMode(backend_cfg.get("context_parallel", ContextParallelMode.ULYSSES))
    sizes = backend_cfg.get("sizes", [DEFAULT_IMAGE_SIZE])
    default_width, default_height = sizes[0]
    default_prompt = backend_cfg.get("prompt", DEFAULT_PROMPT)
    inference_steps = backend_cfg.get("steps", DEFAULT_INFERENCE_STEPS)
    guidance_scale = backend_cfg.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)
    max_sequence_length = backend_cfg.get("max_sequence_length", DEFAULT_MAX_SEQUENCE_LENGTH)

    pipeline = get_pipeline(model_name, multi_gpu=multi_gpu, context_parallel=context_parallel)

    if multi_gpu:
        tuned_model_path = distributed_output_path(tuned_model_path)
    pipeline = ait.load(pipeline, tuned_model_path)

    def mapping(req) -> dict:
        w, h = (int(x) for x in (req.size or f"{default_width}x{default_height}").split("x"))
        return {
            "prompt": req.prompt,
            "height": h,
            "width": w,
            "num_inference_steps": inference_steps,
            "guidance_scale": guidance_scale,
            "max_sequence_length": max_sequence_length,
        }

    request_lock = threading.Lock()

    def generate(
        prompt: str, height: int, width: int, num_inference_steps: int, guidance_scale: float, max_sequence_length: int
    ) -> bytes:
        kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "max_sequence_length": max_sequence_length,
        }
        with request_lock:
            if multi_gpu:
                _broadcast(_GENERATE_COMMAND, kwargs)
            result = _run_generation(pipeline, kwargs)
            buf = io.BytesIO()
            result.images[0].save(buf, format="PNG")
            return buf.getvalue()

    # WAR: JIT warmup
    warmup_kwargs = {
        "prompt": default_prompt,
        "height": default_height,
        "width": default_width,
        "num_inference_steps": 2,
        "guidance_scale": guidance_scale,
        "max_sequence_length": max_sequence_length,
    }
    _run_generation(pipeline, warmup_kwargs)
    _publish_rank_zero_pid()

    if multi_gpu and not is_rank_zero():
        try:
            _run_follower(pipeline)
        finally:
            dist.destroy_process_group()
        return

    config = dyn.DynamoWorkerConfig(
        type="image",
        model_path=model_name,
        mapping=mapping,
    )
    try:
        dyn.dynamo_worker(generate, config)
    finally:
        if multi_gpu:
            with request_lock:
                _broadcast(_STOP_COMMAND)
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
