# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for disk-backed sample storage."""

import os
import shutil
from unittest.mock import call

import pytest
import torch

from aitune.torch.checkpoint.artifact import ArtifactPath
from aitune.torch.module.sample_store import SampleStore


def test_sample_store_loads_fresh_samples_on_demand(tmp_path):
    store = SampleStore.create(tmp_path, "samples")
    store.append(((torch.tensor([1.0]),), {"value": torch.tensor([2.0])}))

    args, kwargs = store[0]
    args[0].add_(10)
    kwargs["value"].add_(10)

    reloaded_args, reloaded_kwargs = store[0]
    torch.testing.assert_close(reloaded_args[0], torch.tensor([1.0]))
    torch.testing.assert_close(reloaded_kwargs["value"], torch.tensor([2.0]))


def test_sample_store_iterator_loads_fresh_samples(tmp_path):
    store = SampleStore.create(tmp_path, "samples")
    store.append(((torch.tensor([1.0]),), {"value": torch.tensor([2.0])}))

    args, kwargs = next(store.iter_samples())
    args[0].add_(10)
    kwargs["value"].add_(10)

    reloaded_args, reloaded_kwargs = next(store.iter_samples())
    torch.testing.assert_close(reloaded_args[0], torch.tensor([1.0]))
    torch.testing.assert_close(reloaded_kwargs["value"], torch.tensor([2.0]))


def test_sample_store_evicts_linux_page_cache(mocker, tmp_path):
    store = SampleStore.create(tmp_path, "samples")
    store.append(((torch.tensor([1.0]),), {}))
    store.append(((torch.tensor([2.0]),), {}))
    mocker.patch("aitune.torch.module.sample_store.sys.platform", "linux")
    mocker.patch("aitune.torch.module.sample_store.os.POSIX_FADV_DONTNEED", 4, create=True)
    open_file = mocker.patch("aitune.torch.module.sample_store.os.open", side_effect=[10, 11])
    fsync = mocker.patch("aitune.torch.module.sample_store.os.fsync")
    fadvise = mocker.patch("aitune.torch.module.sample_store.os.posix_fadvise", create=True)
    close = mocker.patch("aitune.torch.module.sample_store.os.close")

    store.evict_page_cache()

    assert open_file.call_args_list == [
        call(tmp_path / "samples" / "sample-00000.pt", os.O_RDWR),
        call(tmp_path / "samples" / "sample-00001.pt", os.O_RDWR),
    ]
    assert fsync.call_args_list == [call(10), call(11)]
    assert fadvise.call_args_list == [call(10, 0, 0, 4), call(11, 0, 0, 4)]
    assert close.call_args_list == [call(10), call(11)]


@pytest.mark.parametrize(
    ("platform", "fadvise", "dontneed"),
    [
        ("darwin", object(), 4),
        ("linux", None, 4),
        ("linux", object(), None),
    ],
)
def test_sample_store_skips_page_cache_eviction_when_unsupported(mocker, tmp_path, platform, fadvise, dontneed):
    store = SampleStore.create(tmp_path, "samples")
    store.append(((torch.tensor([1.0]),), {}))
    mocker.patch("aitune.torch.module.sample_store.sys.platform", platform)
    mocker.patch("aitune.torch.module.sample_store.os.posix_fadvise", fadvise, create=True)
    mocker.patch("aitune.torch.module.sample_store.os.POSIX_FADV_DONTNEED", dontneed, create=True)
    open_file = mocker.patch("aitune.torch.module.sample_store.os.open")

    store.evict_page_cache()

    open_file.assert_not_called()


def test_sample_store_continues_page_cache_eviction_after_failure(mocker, tmp_path):
    store = SampleStore.create(tmp_path, "samples")
    store.append(((torch.tensor([1.0]),), {}))
    store.append(((torch.tensor([2.0]),), {}))
    mocker.patch("aitune.torch.module.sample_store.sys.platform", "linux")
    mocker.patch("aitune.torch.module.sample_store.os.POSIX_FADV_DONTNEED", 4, create=True)
    mocker.patch("aitune.torch.module.sample_store.os.open", side_effect=[10, 11])
    fsync = mocker.patch("aitune.torch.module.sample_store.os.fsync", side_effect=[OSError, None])
    fadvise = mocker.patch("aitune.torch.module.sample_store.os.posix_fadvise", create=True)
    close = mocker.patch("aitune.torch.module.sample_store.os.close")

    store.evict_page_cache()

    assert fsync.call_args_list == [call(10), call(11)]
    fadvise.assert_called_once_with(11, 0, 0, 4)
    assert close.call_args_list == [call(10), call(11)]


def test_sample_store_samples_remain_readable_after_page_cache_eviction(tmp_path):
    store = SampleStore.create(tmp_path, "samples")
    store.append(((torch.tensor([1.0]),), {}))
    store.append(((torch.tensor([2.0]),), {}))

    store.evict_page_cache()

    assert [args[0].item() for args, _ in store] == [1.0, 2.0]


def test_sample_store_create_removes_existing_samples(tmp_path):
    samples_path = tmp_path / "samples"
    samples_path.mkdir()
    torch.save(((torch.tensor([1.0]),), {}), samples_path / "sample-00000.pt")
    torch.save(((torch.tensor([2.0]),), {}), samples_path / "sample-00001.pt")

    store = SampleStore.create(tmp_path, "samples")

    assert len(store) == 0
    assert list(samples_path.iterdir()) == []

    store.append(((torch.tensor([3.0]),), {}))
    assert [path.name for path in samples_path.iterdir()] == ["sample-00000.pt"]


def test_sample_store_checkpoint_state_references_sample_directory(tmp_path):
    store = SampleStore.create(tmp_path, "samples")
    store.append(((torch.tensor([1.0]),), {}))
    store.append(((torch.tensor([2.0]),), {}))

    assert store.to_dict() == {
        "artifact": ArtifactPath(tmp_path, "samples"),
    }
    assert not (tmp_path / "samples" / "manifest.json").exists()


def test_sample_store_restores_from_checkpoint_state(tmp_path):
    store = SampleStore.create(tmp_path, "samples")
    store.append(((torch.tensor([1.0]),), {}))
    restored = SampleStore.from_dict(store.to_dict())

    assert len(restored) == 1
    assert restored[0][0][0].item() == 1.0


def test_sample_store_restores_from_relocated_artifact(tmp_path):
    cache_root = tmp_path / "cache"
    samples_path = cache_root / "samples"
    store = SampleStore.create(cache_root, "samples")
    store.append(((torch.tensor([1.0]),), {}))
    checkpoint_root = tmp_path / "checkpoint"
    relocated_samples = checkpoint_root / "samples"
    shutil.copytree(samples_path, relocated_samples)
    shutil.rmtree(samples_path)

    restored = SampleStore.from_dict({"artifact": ArtifactPath(checkpoint_root, "samples")})

    assert len(restored) == 1
    assert restored[0][0][0].item() == 1.0


def test_sample_store_loads_numbered_files_in_order(tmp_path):
    samples_path = tmp_path / "samples"
    samples_path.mkdir()
    torch.save(((torch.tensor([2.0]),), {}), samples_path / "sample-00001.pt")
    torch.save(((torch.tensor([1.0]),), {}), samples_path / "sample-00000.pt")

    store = SampleStore(ArtifactPath(tmp_path, "samples"))

    assert [args[0].item() for args, _ in store] == [1.0, 2.0]


def test_sample_store_append_continues_after_existing_samples(tmp_path):
    samples_path = tmp_path / "samples"
    samples_path.mkdir()
    torch.save(((torch.tensor([1.0]),), {}), samples_path / "sample-00000.pt")
    torch.save(((torch.tensor([2.0]),), {}), samples_path / "sample-00001.pt")
    store = SampleStore(ArtifactPath(tmp_path, "samples"))

    store.append(((torch.tensor([3.0]),), {}))

    assert (samples_path / "sample-00002.pt").exists()
    assert [args[0].item() for args, _ in store] == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("filename", ["metadata.json", "custom-name.pt", "sample-00001.pt"])
def test_sample_store_rejects_unexpected_directory_entries(tmp_path, filename):
    samples_path = tmp_path / "samples"
    samples_path.mkdir()
    torch.save(((torch.tensor([1.0]),), {}), samples_path / filename)

    with pytest.raises(ValueError, match="must contain only contiguous"):
        SampleStore(ArtifactPath(tmp_path, "samples"))


def test_sample_store_reports_missing_sample_file(tmp_path):
    samples_path = tmp_path / "samples"
    samples_path.mkdir()
    sample_path = samples_path / "sample-00000.pt"
    torch.save(((torch.tensor([1.0]),), {}), sample_path)
    store = SampleStore(ArtifactPath(tmp_path, "samples"))
    sample_path.unlink()

    with pytest.raises(FileNotFoundError):
        store[0]


def test_sample_store_supports_sequence_indexing(tmp_path):
    samples = [((torch.tensor([value]),), {}) for value in [1.0, 2.0, 3.0]]
    store = SampleStore.from_samples(samples, tmp_path, "samples")

    assert store[-1][0][0].item() == 3.0
    assert [args[0].item() for args, _ in store[1:]] == [2.0, 3.0]
    assert [args[0].item() for args, _ in store[::-1]] == [3.0, 2.0, 1.0]


def test_sample_store_supports_empty_sequence(tmp_path):
    store = SampleStore.create(tmp_path, "samples")

    assert len(store) == 0
    assert list(store) == []
    assert store[:] == []


def test_sample_store_remaps_only_cuda_storage(mocker):
    restore = mocker.patch("aitune.torch.module.sample_store.torch.serialization.default_restore_location")
    restore.return_value = "remapped"
    map_location = SampleStore._map_location(torch.device("cuda", 1))
    cpu_storage = object()
    cuda_storage = object()

    assert map_location(cpu_storage, "cpu") is cpu_storage
    assert map_location(cuda_storage, "cuda:0") == "remapped"
    restore.assert_called_once_with(cuda_storage, "cuda:1")
