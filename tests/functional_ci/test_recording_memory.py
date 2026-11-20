# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recording samples for multiple modules keeps GPU memory bounded."""

import gc
from pathlib import Path

import pytest
import torch

from aitune.torch.module.recording_module import RecordingModule

NUM_MODULES = 6
SAMPLE_SIZE_BYTES = 8 * 1024 * 1024


@pytest.mark.matrix("other.multi-module-recording-memory")
def test_multi_module_recording_does_not_retain_gpu_samples(tmp_path: Path, device: torch.device):
    """Persisted samples do not keep a GPU-sized tensor alive per module."""
    sample = torch.ones(SAMPLE_SIZE_BYTES // torch.float32.itemsize, device=device)
    recorders = [
        RecordingModule(
            torch.nn.Identity(), f"memory-test-{index}", cache_dir_resolver=lambda d=tmp_path / str(index): d
        )
        for index in range(NUM_MODULES)
    ]

    gc.collect()
    torch.cuda.synchronize(device)
    baseline_memory = torch.cuda.memory_allocated(device)
    with torch.no_grad():
        for recorder in recorders:
            recorder(sample)
    gc.collect()
    torch.cuda.synchronize(device)
    retained_memory = torch.cuda.memory_allocated(device) - baseline_memory

    for recorder in recorders:
        assert len(recorder.samples_for_graph_spec(recorder.graph_specs[0])) == 1
    assert retained_memory < SAMPLE_SIZE_BYTES * 2, (
        f"Recording {NUM_MODULES} modules retained {retained_memory / 1024**2:.1f} MiB of GPU memory; "
        "expected disk-backed samples to keep growth below two sample sizes"
    )
