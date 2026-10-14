# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage Schnell checkpoint weights on CPU before backend deployment."""

import json
from pathlib import Path

import torch

import aitune.torch as ait
from aitune.torch.checkpoint.local_torch_storage import LocalTorchStorage
from aitune.torch.checkpoint.storage_tasks import STATE_DICT_FILE, TorchLoadTask

from .config import Recipe
from .distributed import distributed_output_path, is_rank_zero, synchronize


class _CpuLoadTask(TorchLoadTask):
    """Avoid a second copy of checkpoint weights on the GPU during loading."""

    def load(self, path: Path, state_dict: dict | None = None) -> dict:
        # AITune checkpoints include backend objects, so retain its weights_only=False
        # behavior. Backends restore their saved target devices during deployment.
        return torch.load(path / STATE_DICT_FILE, map_location="cpu", weights_only=False)


def cpu_checkpoint_storage() -> LocalTorchStorage:
    """Keep standard extraction, hash verification, and artifact relocation."""
    storage = LocalTorchStorage()
    storage.load_tasks = [_CpuLoadTask() if isinstance(task, TorchLoadTask) else task for task in storage.load_tasks]
    return storage


def save_checkpoint(pipeline, recipe: Recipe, tuning_seconds: float) -> None:
    """Save one archive per rank, then publish the completed run's manifest."""
    manifest_path = recipe.artifact_path.with_suffix(".manifest.json")
    if is_rank_zero():
        manifest_path.unlink(missing_ok=True)
    synchronize()
    ait.save(pipeline, distributed_output_path(recipe.artifact_path))
    synchronize()
    if is_rank_zero():
        artifacts = [
            distributed_output_path(recipe.artifact_path, rank, recipe.gpu_count) for rank in range(recipe.gpu_count)
        ]
        if not all(path.is_file() for path in artifacts):
            raise RuntimeError("Not every rank saved its checkpoint")
        manifest = {
            "model": recipe["model"],
            "execution": recipe["execution"],
            "workload": recipe["workload"],
            "tuning": recipe["tuning"],
            "samples": recipe.samples("tune"),
            "tuning_seconds": tuning_seconds,
            "artifacts": [path.name for path in artifacts],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    synchronize()


def check_checkpoint(recipe: Recipe) -> dict:
    manifest = json.loads(recipe.artifact_path.with_suffix(".manifest.json").read_text())
    for key in ("model", "execution", "workload", "tuning"):
        if manifest[key] != recipe[key]:
            raise ValueError(f"Checkpoint {key} differs from the selected recipe; tune this recipe first")
    for rank in range(recipe.gpu_count):
        path = distributed_output_path(recipe.artifact_path, rank, recipe.gpu_count)
        if not path.is_file():
            raise FileNotFoundError(path)
    return manifest


def load_checkpoint(pipeline, recipe: Recipe):
    check_checkpoint(recipe)
    return ait.load(pipeline, distributed_output_path(recipe.artifact_path), storage=cpu_checkpoint_storage())
