# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schnell loading and seeded end-to-end Diffusers inference."""

from __future__ import annotations

import logging
import os
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import Config, InputRecord, Prompt

if TYPE_CHECKING:
    from diffusers import FluxPipeline
    from PIL.Image import Image

    from aitune.torch.checkpoint.local_torch_storage import LocalTorchStorage

logger = logging.getLogger(__name__)


def check_gpu(cfg: Config) -> None:
    import torch
    import torch.distributed as dist

    count = cfg["execution"]["gpu_count"]
    if int(os.environ.get("WORLD_SIZE", "1")) != count or int(os.environ.get("LOCAL_WORLD_SIZE", "1")) != count:
        raise ValueError("Launcher world size differs from execution.gpu_count; use the recipe CLI")
    if not torch.cuda.is_available() or torch.cuda.device_count() != count:
        raise ValueError(f"Expose exactly {count} CUDA GPU(s)")
    if count > 1 and (not dist.is_initialized() or dist.get_world_size() != count):
        raise ValueError("Context parallelism requires an initialized NCCL process group")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if not torch.cuda.is_bf16_supported():
        raise ValueError("This recipe requires native BF16 support")
    torch.backends.cuda.matmul.allow_tf32 = False


def load_model(
    args: Namespace, cfg: Config, variant: str = "original", source: dict[str, Any] | None = None
) -> FluxPipeline:
    import torch
    from diffusers import FluxPipeline

    from .records import artifact_metadata, contract, source_identity

    check_gpu(cfg)
    source = source or source_identity(args, cfg)
    metadata = None
    if variant == "aitune":
        metadata = artifact_metadata(cfg)
        if metadata["contract"] != contract(cfg, source):
            raise ValueError("Source, precision, execution, or workload differs from the tuning artifact; re-tune")
    kwargs = {"torch_dtype": torch.bfloat16, "use_safetensors": True}
    if args.model_id:
        kwargs["revision"] = cfg["model"]["revision"]
    logger.info("Loading base checkpoint")
    pipe = FluxPipeline.from_pretrained(args.model_id or args.checkpoint, **kwargs)
    if pipe.transformer.config.guidance_embeds:
        raise ValueError("This recipe requires a Schnell transformer without guidance embeddings")
    logger.info("Moving pipeline to CUDA")
    pipe = pipe.to("cuda")
    if cfg["execution"]["gpu_count"] > 1:
        from diffusers import ContextParallelConfig

        if pipe.transformer.config.num_attention_heads % cfg["execution"]["gpu_count"]:
            raise ValueError("Transformer attention heads must be divisible by the Ulysses degree")
        pipe.transformer.set_attention_backend("_native_cudnn")
        pipe.transformer.enable_parallelism(config=ContextParallelConfig(ulysses_degree=cfg["execution"]["gpu_count"]))
    pipe.set_progress_bar_config(disable=True)
    if metadata:
        import aitune.torch as ait

        check_environment(metadata["environment"])
        logger.info("Restoring AITune artifact: %s", cfg.artifact)
        pipe = ait.load(pipe, cfg.artifact, storage=cpu_staging_storage())
    return pipe


def generate(pipe: FluxPipeline, cfg: Config, item: Prompt | InputRecord) -> Image:
    import torch

    with torch.inference_mode():
        image = pipe(
            prompt=item["prompt"],
            **cfg["workload"],
            generator=torch.Generator("cpu").manual_seed(item["seed"]),
            num_images_per_prompt=1,
            output_type="pil",
        ).images[0]
    return image.convert("RGB")


def synchronize() -> None:
    import torch

    torch.cuda.synchronize()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def save_images(pipe: FluxPipeline, cfg: Config, records: Sequence[InputRecord], directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for index, item in enumerate(records, 1):
        logger.info("Generating image %d/%d: %s", index, len(records), item["id"])
        generate(pipe, cfg, item).save(directory / f"{item['id']}.png")
    logger.info("Saved %d images to %s", len(records), directory)


def cpu_staging_storage() -> LocalTorchStorage:
    """Avoid duplicating CUDA weights when restoring a saved pipeline."""
    import torch

    from aitune.torch.checkpoint.local_torch_storage import LocalTorchStorage
    from aitune.torch.checkpoint.storage_tasks import STATE_DICT_FILE, TorchLoadTask

    class CpuTorchLoadTask(TorchLoadTask):
        def load(self, path: Path, state_dict: dict[str, Any] | None = None) -> dict[str, Any]:
            return torch.load(path / STATE_DICT_FILE, map_location="cpu", weights_only=False)

    storage = LocalTorchStorage()
    storage.load_tasks = [
        CpuTorchLoadTask() if isinstance(task, TorchLoadTask) else task for task in storage.load_tasks
    ]
    return storage


def check_environment(built: dict[str, Any]) -> None:
    from .records import environment

    current = environment()
    if current["gpu"]["compute_capability"] != built["gpu"]["compute_capability"]:
        raise ValueError("GPU architecture differs from tuning; create a new artifact")
    for name in ("aitune", "torch", "tensorrt", "torch-tensorrt", "diffusers", "transformers", "torchao"):
        if current["packages"][name] != built["packages"][name]:
            raise ValueError(f"{name} differs from the artifact build environment")
