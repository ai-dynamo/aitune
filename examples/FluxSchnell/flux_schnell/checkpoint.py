# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage Schnell checkpoint weights on CPU before backend deployment."""

from pathlib import Path

import torch

from aitune.torch.checkpoint.local_torch_storage import LocalTorchStorage
from aitune.torch.checkpoint.storage_tasks import STATE_DICT_FILE, TorchLoadTask


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
