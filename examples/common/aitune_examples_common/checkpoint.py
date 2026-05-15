# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint relocation helpers for examples."""

import shutil
from pathlib import Path

AIT_EXTENSION = ".ait"
CHECKPOINTS_DIR = Path("checkpoints")
RELOCATED_CHECKPOINT_DIR = Path("/tmp")
SHA_TYPE = "256"


def relocated_checkpoint_path(checkpoint_path: str | Path, target_dir: str | Path = RELOCATED_CHECKPOINT_DIR) -> Path:
    """Return the load path for an example checkpoint copied to target_dir."""
    return Path(target_dir) / Path(checkpoint_path).name


def copy_checkpoint_to_tmp(checkpoint_path: str | Path, target_dir: str | Path = RELOCATED_CHECKPOINT_DIR) -> Path:
    """Copy a normally saved example checkpoint archive to target_dir.

    The returned path is the path that should be passed to ``aitune.torch.load``.
    """
    source_root = _checkpoint_root(Path(checkpoint_path))
    target_path = relocated_checkpoint_path(checkpoint_path, target_dir)
    target_root = _checkpoint_root(target_path)

    source_archive_path = _archive_path(source_root)
    source_sha_path = _sha_sums_path(source_root)
    target_archive_path = _archive_path(target_root)
    target_sha_path = _sha_sums_path(target_root)

    if not source_archive_path.is_file():
        raise FileNotFoundError(f"Checkpoint archive not found: {source_archive_path}")
    if not source_sha_path.is_file():
        raise FileNotFoundError(f"Checkpoint SHA sidecar not found: {source_sha_path}")

    target_root.parent.mkdir(parents=True, exist_ok=True)
    target_archive_path.unlink(missing_ok=True)
    target_sha_path.unlink(missing_ok=True)
    shutil.rmtree(target_root, ignore_errors=True)

    shutil.copy2(source_archive_path, target_archive_path)
    shutil.copy2(source_sha_path, target_sha_path)
    return target_path


def _checkpoint_root(path: Path) -> Path:
    if path.suffix == AIT_EXTENSION:
        path = path.with_suffix("")
    if path.is_absolute():
        return path
    return CHECKPOINTS_DIR / path


def _archive_path(checkpoint_root: Path) -> Path:
    return checkpoint_root.with_name(checkpoint_root.name + AIT_EXTENSION)


def _sha_sums_path(checkpoint_root: Path) -> Path:
    return checkpoint_root.parent / f"{checkpoint_root.stem}_sha{SHA_TYPE}_sums.txt"
