# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for model state storage functionality."""

from pathlib import Path

import pytest

from aitune.torch.checkpoint.local_torch_storage import LocalTorchStorage
from aitune.torch.checkpoint.storage_tasks import (
    AIT_EXTENSION,
    get_sha_sums_path,
)


def do_test_save_load_torch_task(storage, checkpoint_path, compress_checkpoint=True, sha_type="256"):
    state_dict = {"a": 1, "b": 2}

    storage.save(checkpoint_path, state_dict)

    if compress_checkpoint:
        assert (checkpoint_path.with_name(checkpoint_path.name + AIT_EXTENSION)).exists()
        internal_sha_sums = get_sha_sums_path(checkpoint_path, sha_type)
        assert internal_sha_sums.exists()
        external_sha_sums = internal_sha_sums.parent.parent / internal_sha_sums.name
        assert external_sha_sums.exists()
    else:
        assert checkpoint_path.exists()

    loaded_state_dict = storage.load(checkpoint_path)
    assert loaded_state_dict == state_dict


def test_save_load_torch_task_defaults(tmp_path):
    checkpoint_path = tmp_path / "test"
    do_test_save_load_torch_task(LocalTorchStorage(), checkpoint_path)


def test_save_load_torch_task_custom_base_path(tmp_path):
    checkpoint_path = Path("test")
    do_test_save_load_torch_task(LocalTorchStorage(base_folder=tmp_path), tmp_path / checkpoint_path)


def test_save_load_torch_task_custom_extension(tmp_path):
    checkpoint_path = tmp_path / "test.my_extension"
    do_test_save_load_torch_task(LocalTorchStorage(base_folder=checkpoint_path), checkpoint_path)


def test_save_load_torch_task_without_compression(tmp_path):
    checkpoint_path = tmp_path / "test"
    do_test_save_load_torch_task(
        LocalTorchStorage(compress_checkpoint=False), checkpoint_path, compress_checkpoint=False
    )


@pytest.mark.parametrize("sha_type", ["256", "512"])
def test_save_load_torch_task_sha(tmp_path, sha_type):
    """Test LocalTorchStorage with SHA-512."""
    checkpoint_path = tmp_path / "test"
    do_test_save_load_torch_task(
        LocalTorchStorage(sha_type=sha_type),
        checkpoint_path,
        sha_type=sha_type,
    )
