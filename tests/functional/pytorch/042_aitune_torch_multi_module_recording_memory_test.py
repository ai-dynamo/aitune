# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# scope = "always"
# ///

"""Verify that recording samples for multiple modules has bounded GPU memory usage."""

import gc
import tempfile
from pathlib import Path

import torch

from aitune.torch.module.recording_module import RecordingModule

_NUM_MODULES = 6
_SAMPLE_SIZE_BYTES = 8 * 1024 * 1024


def test_multi_module_recording_does_not_retain_gpu_samples(tmp_path):
    """Persisted samples do not keep a GPU-sized tensor alive per module."""
    device = torch.device("cuda")
    sample = torch.ones(_SAMPLE_SIZE_BYTES // torch.float32.itemsize, device=device)

    def module_cache_dir_resolver(index):
        return lambda: tmp_path / f"module-{index}"

    recorders = [
        RecordingModule(
            torch.nn.Identity(),
            f"memory-test-{index}",
            cache_dir_resolver=module_cache_dir_resolver(index),
        )
        for index in range(_NUM_MODULES)
    ]

    gc.collect()
    torch.cuda.synchronize(device)
    baseline_memory = torch.cuda.memory_allocated(device)

    with torch.no_grad():
        for recorder in recorders:
            recorder(sample)

    gc.collect()
    torch.cuda.synchronize(device)
    recorded_memory = torch.cuda.memory_allocated(device)

    for recorder in recorders:
        graph_spec = recorder.graph_specs[0]
        assert len(recorder.samples_for_graph_spec(graph_spec)) == 1

    retained_memory = recorded_memory - baseline_memory
    assert retained_memory < _SAMPLE_SIZE_BYTES * 2, (
        f"Recording {_NUM_MODULES} modules retained {retained_memory / 1024**2:.1f} MiB of GPU memory; "
        "expected disk-backed samples to keep growth below two sample sizes"
    )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as cache_dir:
        test_multi_module_recording_does_not_retain_gpu_samples(Path(cache_dir))
