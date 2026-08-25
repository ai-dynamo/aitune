# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for disk-backed sample storage."""

import copy
import gc
import tempfile
from pathlib import Path

import torch

from aitune.torch.module.sample_store import SampleStore, ensure_sample_store


def test_sample_store_loads_fresh_samples_on_demand(tmp_path):
    store = SampleStore.create(tmp_path, "samples")
    store.append(((torch.tensor([1.0]),), {"value": torch.tensor([2.0])}))

    args, kwargs = store[0]
    args[0].add_(10)
    kwargs["value"].add_(10)

    reloaded_args, reloaded_kwargs = store[0]
    torch.testing.assert_close(reloaded_args[0], torch.tensor([1.0]))
    torch.testing.assert_close(reloaded_kwargs["value"], torch.tensor([2.0]))


def test_sample_store_deepcopy_shares_persisted_files(tmp_path):
    store = SampleStore.create(tmp_path, "samples")
    store.append(((torch.tensor([1.0]),), {}))

    assert copy.deepcopy(store) is store


def test_ensure_sample_store_reuses_existing_store(tmp_path):
    store = SampleStore.create(tmp_path, "samples")

    assert ensure_sample_store(store, tmp_path / "backend") is store
    assert not (tmp_path / "backend").exists()


def test_sample_store_keeps_temporary_root_alive():
    owner = tempfile.TemporaryDirectory()
    root = Path(owner.name)
    store = SampleStore.create(root, "samples", owner=owner)
    store.append(((torch.tensor([1.0]),), {}))

    del owner
    gc.collect()

    assert store.artifact.path.exists()


def test_sample_store_remaps_only_cuda_storage(mocker):
    restore = mocker.patch("aitune.torch.module.sample_store.torch.serialization.default_restore_location")
    restore.return_value = "remapped"
    map_location = SampleStore._map_location(torch.device("cuda", 1))
    cpu_storage = object()
    cuda_storage = object()

    assert map_location(cpu_storage, "cpu") is cpu_storage
    assert map_location(cuda_storage, "cuda:0") == "remapped"
    restore.assert_called_once_with(cuda_storage, "cuda:1")
