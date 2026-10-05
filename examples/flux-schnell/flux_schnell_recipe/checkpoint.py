# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage checkpoint tensors in host memory before AITune deploys the backends."""

import torch
from aitune.torch.checkpoint.local_torch_storage import LocalTorchStorage
from aitune.torch.checkpoint.storage_tasks import STATE_DICT_FILE, TorchLoadTask


class CpuTorchLoadTask(TorchLoadTask):
    def load(self, path, state_dict=None):
        # AITune 0.6.0 otherwise restores tensors to their saved CUDA device,
        # duplicating the live pipeline's weights before copying them into it.
        # As in AITune's loader, backend objects require trusted pickle input.
        return torch.load(path / STATE_DICT_FILE, map_location="cpu", weights_only=False)


def cpu_staging_storage():
    storage = LocalTorchStorage()
    # Preserve extraction, checksum verification and backend artifact relocation.
    storage.load_tasks = [CpuTorchLoadTask() if isinstance(task, TorchLoadTask) else task
                          for task in storage.load_tasks]
    return storage
