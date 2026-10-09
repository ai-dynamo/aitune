# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-node NCCL ownership and rank-local paths for the FLUX recipe."""

from dataclasses import replace
from datetime import timedelta
import os


def rank_config(cfg, rank):
    count = cfg["execution"]["gpu_count"]
    path = cfg.artifact
    data = dict(cfg.data, artifact_path=str(path.with_name(f"{path.stem}.rank-{rank}-of-{count}{path.suffix}")))
    return replace(cfg, data=data)


def initialize(cfg):
    import torch
    import torch.distributed as dist
    count = cfg["execution"]["gpu_count"]
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    rank, local_rank = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    if count != 2 or world != count or local_world != count or rank != local_rank:
        raise ValueError("Use one torchrun node with exactly two local ranks for this recipe")
    if not torch.cuda.is_available() or torch.cuda.device_count() != count:
        raise ValueError("Both ranks must see exactly two CUDA GPUs")
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():
        raise ValueError("Both GPUs must support BF16")
    dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=torch.device("cuda", local_rank))
    devices = collect({"name": torch.cuda.get_device_name(),
                       "capability": torch.cuda.get_device_capability(),
                       "memory": torch.cuda.get_device_properties(local_rank).total_memory})
    if any(device != devices[0] for device in devices):
        raise ValueError("Use matching GPU models, compute capabilities, and VRAM capacities")
    return rank


def collect(value):
    import torch.distributed as dist
    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, value)
    return result


def shutdown():
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    """User-run topology smoke check, before loading the model or tuning."""
    import argparse
    import torch
    import torch.distributed as dist
    from .config import load_config
    from .records import environment, new_run, write_json
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    try:
        rank = initialize(cfg)
        count = cfg["execution"]["gpu_count"]
        source = torch.full((count,), rank, dtype=torch.float32, device="cuda")
        target = torch.empty_like(source)
        dist.all_to_all_single(target, source)
        if not torch.equal(target, torch.arange(count, dtype=torch.float32, device="cuda")):
            raise RuntimeError("NCCL all-to-all produced incorrect data")
        dist.all_reduce(source)
        if not torch.all(source == sum(range(count))):
            raise RuntimeError("NCCL all-reduce produced incorrect data")
        environments = collect(environment())
        if rank == 0:
            directory = new_run(cfg, "preflight")
            write_json(directory / "report.json", {"passed": True, "gpu_count": count,
                       "collectives": ["all_to_all_single", "all_reduce"], "environments": environments})
            print(directory / "report.json", flush=True)
    finally:
        shutdown()


if __name__ == "__main__":
    main()
