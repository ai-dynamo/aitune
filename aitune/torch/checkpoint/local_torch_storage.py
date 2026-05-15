# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local torch storage class which saves torch checkpoint on local disk."""

from pathlib import Path
from typing import Literal

from aitune.torch.checkpoint.storage import Storage
from aitune.torch.checkpoint.storage_tasks import (
    CopyBackendArtifactsTask,
    MakeFolderTask,
    RelocateBackendArtifactsTask,
    RemoveFolderTask,
    ShaSumsLoadTask,
    ShaSumsSaveTask,
    TorchLoadTask,
    TorchSaveTask,
    UnzipLoadTask,
    ZipSaveTask,
)


class LocalTorchStorage(Storage):
    """Implementation of storage that saves/loads from host machine."""

    def __init__(
        self,
        base_folder: str | Path = "checkpoints",
        compress_checkpoint: bool = True,
        overwrite_checkpoint_on_tune: bool = True,
        remove_checkpoint_after_tune: bool = False,
        sha_type: Literal["256", "512"] = "256",
    ):
        """Initialize the storage.

        Args:
            base_folder: The base folder to store the checkpoints.
            compress_checkpoint: Whether to compress the checkpoint folder. Compressing also performs sha hash of the checkpoint folder.
            overwrite_checkpoint_on_tune: Whether to overwrite the checkpoint folder on tuning. Otherwise raises exception
                if the checkpoint folder already exists.
            remove_checkpoint_after_tune: Whether to remove the checkpoint folder after tuning.
            sha_type: The SHA type to use for hashing. Must be either "256" or "512".

        Note: if the checkpoint folder is removed, it will be recreated from ai tune checkpoint at first load.
        """
        super().__init__(
            save_tasks=[
                MakeFolderTask(overwrite=overwrite_checkpoint_on_tune),
                CopyBackendArtifactsTask(),
                TorchSaveTask(),
                ShaSumsSaveTask(sha_type=sha_type) if compress_checkpoint else None,
                ZipSaveTask() if compress_checkpoint else None,
                RemoveFolderTask() if remove_checkpoint_after_tune else None,
            ],
            load_tasks=[
                UnzipLoadTask() if compress_checkpoint else None,
                ShaSumsLoadTask(sha_type=sha_type) if compress_checkpoint else None,
                TorchLoadTask(),
                RelocateBackendArtifactsTask(),
            ],
        )
        self.base_folder = Path(base_folder)

    def _get_target_folder_path(self, path: str | Path) -> Path:
        """Get the target folder path."""
        path = super()._get_target_folder_path(path)
        return self.base_folder / path
