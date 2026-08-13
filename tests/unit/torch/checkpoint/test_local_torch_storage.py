# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for model state storage functionality."""

import shutil
from pathlib import Path

import pytest
import torch

from aitune.torch.backend import ArtifactPath
from aitune.torch.checkpoint.local_torch_storage import LocalTorchStorage
from aitune.torch.checkpoint.storage_tasks import (
    AIT_EXTENSION,
    get_sha_sums_path,
)
from tests.toy_models.torch_models import ToyTorchModel
from tests.unit.torch.checkpoint.helpers import _backend_state_with_paths, _only_backend_data


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


def test_save_copy_load_relocates_backend_artifact_paths(tmp_path):
    compile_root = tmp_path / "compile"
    serve_root = tmp_path / "serve"
    artifact_root = tmp_path / "artifacts"
    compile_root.mkdir()
    serve_root.mkdir()
    artifact_root.mkdir()
    artifact_path = artifact_root / "model.plan"
    artifact_path.write_bytes(b"engine")

    state_dict = _backend_state_with_paths(
        engine_path=ArtifactPath(root=artifact_root, relative_path=Path("model.plan"))
    )
    source_checkpoint = Path("my_model")
    LocalTorchStorage(base_folder=compile_root).save(source_checkpoint, state_dict)

    shutil.copy2(compile_root / "my_model.ait", serve_root / "my_model.ait")
    shutil.rmtree(compile_root)
    shutil.rmtree(artifact_root)

    loaded_state_dict = LocalTorchStorage(base_folder=serve_root).load(source_checkpoint)

    loaded_artifact = _only_backend_data(loaded_state_dict)["engine_path"]
    assert loaded_artifact == ArtifactPath(
        root=(serve_root / "my_model").resolve(),
        relative_path=Path("1/1/model.plan"),
    )
    assert loaded_artifact.path.read_bytes() == b"engine"


def test_save_move_load_relocates_backend_artifact_paths(tmp_path):
    compile_root = tmp_path / "compile"
    serve_root = tmp_path / "serve"
    artifact_root = tmp_path / "artifacts"
    compile_root.mkdir()
    serve_root.mkdir()
    artifact_root.mkdir()
    artifact_path = artifact_root / "model.plan"
    artifact_path.write_bytes(b"engine")

    state_dict = _backend_state_with_paths(
        engine_path=ArtifactPath(root=artifact_root, relative_path=Path("model.plan"))
    )
    LocalTorchStorage(base_folder=compile_root).save(Path("my_model"), state_dict)

    shutil.move(str(compile_root / "my_model.ait"), serve_root / "moved_model.ait")
    shutil.rmtree(compile_root)
    shutil.rmtree(artifact_root)

    loaded_state_dict = LocalTorchStorage(base_folder=serve_root).load(Path("moved_model"))

    loaded_artifact = _only_backend_data(loaded_state_dict)["engine_path"]
    assert loaded_artifact == ArtifactPath(
        root=(serve_root / "moved_model").resolve(),
        relative_path=Path("1/1/model.plan"),
    )
    assert loaded_artifact.path.read_bytes() == b"engine"


def test_save_copy_load_torch_model_preserves_outputs(tmp_path):
    compile_root = tmp_path / "compile"
    serve_root = tmp_path / "serve"
    compile_root.mkdir()
    serve_root.mkdir()

    source_model = ToyTorchModel(is_linear=True).eval()
    sample = source_model.sample()
    with torch.no_grad():
        expected = source_model(sample)

    checkpoint_name = Path("toy_model")
    LocalTorchStorage(base_folder=compile_root).save(checkpoint_name, source_model.state_dict())

    shutil.copy2(compile_root / "toy_model.ait", serve_root / "toy_model.ait")

    restored_model = ToyTorchModel(is_linear=True).eval()
    restored_state_dict = LocalTorchStorage(base_folder=serve_root).load(checkpoint_name)
    restored_model.load_state_dict(restored_state_dict)

    with torch.no_grad():
        actual = restored_model(sample)

    torch.testing.assert_close(actual, expected)


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
