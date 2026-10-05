# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schnell loading and seeded end-to-end Diffusers inference."""

import os


def check_gpu():
    import torch
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or int(os.environ.get("LOCAL_WORLD_SIZE", "1")) != 1:
        raise ValueError("This recipe requires one process; do not use torchrun")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one CUDA GPU, for example CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("This recipe requires native BF16 support")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False


def load_model(args, cfg, variant="original", source=None):
    import torch
    from diffusers import FluxPipeline
    from .records import source_identity, contract, artifact_metadata
    check_gpu()
    source = source or source_identity(args, cfg)
    metadata = None
    if variant == "aitune":
        metadata = artifact_metadata(cfg)
        if metadata["contract"] != contract(cfg, source):
            raise ValueError("Source, precision, execution, or workload differs from the tuning artifact; re-tune")
    kwargs = {"torch_dtype": torch.bfloat16, "use_safetensors": True}
    if args.model_id:
        kwargs["revision"] = cfg["model"]["revision"]
    pipe = FluxPipeline.from_pretrained(args.model_id or args.checkpoint, **kwargs)
    if pipe.transformer.config.guidance_embeds:
        raise ValueError("This recipe requires a Schnell transformer without guidance embeddings")
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    if metadata:
        import aitune.torch as ait
        from .checkpoint import cpu_staging_storage
        from .records import environment
        current = environment()
        built = metadata["environment"]
        if current["gpu"]["compute_capability"] != built["gpu"]["compute_capability"]:
            raise ValueError("GPU architecture differs from tuning; create a new artifact")
        for name in ("aitune", "torch", "tensorrt", "torch-tensorrt", "diffusers", "transformers", "torchao"):
            if current["packages"][name] != built["packages"][name]:
                raise ValueError(f"{name} differs from the artifact build environment")
        pipe = ait.load(pipe, cfg.artifact, storage=cpu_staging_storage())
    return pipe


def generate(pipe, cfg, item):
    import torch
    with torch.inference_mode():
        image = pipe(prompt=item["prompt"], **cfg["workload"],
                     generator=torch.Generator("cpu").manual_seed(item["seed"]),
                     num_images_per_prompt=1, output_type="pil").images[0]
    return image.convert("RGB")


def synchronize():
    import torch
    torch.cuda.synchronize()
