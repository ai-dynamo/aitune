# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for model state storage functionality."""

import shutil

import pytest

from aitune.torch.checkpoint.storage_tasks import (
    AIT_EXTENSION,
    MakeFolderTask,
    ShaSumsLoadTask,
    ShaSumsSaveTask,
    TorchLoadTask,
    TorchSaveTask,
    UnzipLoadTask,
    ZipSaveTask,
    check_checkpoint_valid,
    get_sha_sums_path,
)
from tests.toy_models.torch_models import ToyTorchModel


def test_save_load_torch_task(tmp_path):
    model = ToyTorchModel(is_linear=True)
    save_task = TorchSaveTask()
    load_task = TorchLoadTask()

    orig_state_dict = model.state_dict()
    orig_state_dict["extra_data"] = "for testing"

    save_task.save(tmp_path, orig_state_dict)
    state_dict = load_task.load(tmp_path)

    assert state_dict.keys() == orig_state_dict.keys()
    assert state_dict["extra_data"] == orig_state_dict["extra_data"]


def test_make_folder_task_overwrite(tmp_path):
    """Test MakeFolderTask with overwrite=True."""
    folder_path = tmp_path / "test_folder"

    # Create the folder task with overwrite=True
    make_folder_task = MakeFolderTask(overwrite=True)

    # Test creating a new folder
    make_folder_task.save(folder_path, {})
    assert folder_path.exists()
    assert folder_path.is_dir()

    # Test that trying to create the same folder again overwrites it
    make_folder_task.save(folder_path, {})
    assert folder_path.exists()
    assert folder_path.is_dir()


def test_make_folder_task_no_overwrite(tmp_path):
    """Test MakeFolderTask with overwrite=False."""
    folder_path = tmp_path / "test_folder"

    # Create the folder task with overwrite=False
    make_folder_task = MakeFolderTask(overwrite=False)

    # Test creating a new folder
    make_folder_task.save(folder_path, {})
    assert folder_path.exists()
    assert folder_path.is_dir()

    # Test that trying to create the same folder again raises FileExistsError
    with pytest.raises(FileExistsError, match=f"Folder {folder_path} already exists"):
        make_folder_task.save(folder_path, {})


@pytest.mark.parametrize("sha_type", ["256", "512"])
def test_sha_sums_save_task(tmp_path, sha_type):
    # Create test files
    test_file1 = tmp_path / "test_file1.txt"
    test_file2 = tmp_path / "test_file2.txt"
    test_file1.write_text("content1")
    test_file2.write_text("content2")

    sha_sums_save_task = ShaSumsSaveTask(sha_type=sha_type)
    sha_sums_save_task.save(tmp_path, {})

    # Check that sha_sums.txt was created (default)
    sha_sums_file = get_sha_sums_path(tmp_path, sha_type)
    assert sha_sums_file.exists()

    sha_sums_load_task = ShaSumsLoadTask(sha_type=sha_type)
    result = sha_sums_load_task.load(tmp_path)

    # Should return empty dict on success
    assert result == {}

    # modify content of a file
    test_file1.write_text("modified_content")

    with pytest.raises(ValueError, match="Failed to verify SHA hashes for files"):
        sha_sums_load_task.load(tmp_path)

    # remove sha_sums.txt
    sha_sums_file.unlink()

    with pytest.raises(FileNotFoundError, match="SHA hash file not found"):
        sha_sums_load_task.load(tmp_path)


@pytest.mark.parametrize("checkpoint", ["valid", "invalid", "missing"])
def test_zip_unzip_tasks(tmp_path, checkpoint):
    """Test ZipSaveTask and UnzipLoadTask functionality."""
    # Create test files and subdirectories
    test_file1 = tmp_path / "test_file1.txt"
    test_file2 = tmp_path / "test_file2.txt"
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    test_file3 = subdir / "test_file3.txt"

    test_file1.write_text("content1")
    test_file2.write_text("content2")
    test_file3.write_text("content3")

    # Sha256 file should be copied aside zip
    sha_sums = ShaSumsSaveTask()
    sha_sums.save(tmp_path, {})

    # Test zip save task
    zip_save_task = ZipSaveTask()
    zip_save_task.save(tmp_path, {})

    # Check that zip file was created
    zip_file = tmp_path.with_suffix(AIT_EXTENSION)
    assert zip_file.exists()
    sha_sums_file = get_sha_sums_path(tmp_path, "256")
    assert sha_sums_file.exists()

    if checkpoint == "missing":
        # remove completely
        shutil.rmtree(tmp_path)  # noqa: F821
    elif checkpoint == "invalid":
        # remove one file
        test_file1.unlink()  # noqa: F821
        # modify sha_sums file
        sha_sums_file.write_text("invalid_sha_sums")
    else:
        pass  # leave checkpoint as is

    # Test unzip load task
    unzip_load_task = UnzipLoadTask()
    result = unzip_load_task.load(tmp_path)

    # Should return empty dict on success
    assert result == {}

    # Check that files were extracted correctly
    extract_path = tmp_path.with_suffix("")
    assert extract_path.exists()
    assert extract_path.is_dir()

    # Verify all files are present and have correct content
    extracted_file1 = extract_path / "test_file1.txt"
    extracted_file2 = extract_path / "test_file2.txt"
    extracted_subdir = extract_path / "subdir"
    extracted_file3 = extracted_subdir / "test_file3.txt"

    assert extracted_file1.exists()
    assert extracted_file2.exists()
    assert extracted_subdir.exists()
    assert extracted_file3.exists()

    assert extracted_file1.read_text() == "content1"
    assert extracted_file2.read_text() == "content2"
    assert extracted_file3.read_text() == "content3"

    # Test that non-existent zip file raises FileNotFoundError
    non_existent_path = tmp_path / "non_existent"
    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        unzip_load_task.load(non_existent_path)


def test_check_checkpoint_valid_nonexistent_path(tmp_path):
    """Test check_checkpoint_valid with a path that doesn't exist."""
    nonexistent_path = tmp_path / "nonexistent_checkpoint"

    result = check_checkpoint_valid(nonexistent_path)
    assert result is False


def test_check_checkpoint_valid_no_sha_sums_files(tmp_path):
    """Test check_checkpoint_valid with no sha_sums files."""
    # Create a directory but no sha files
    test_dir = tmp_path / "test_checkpoint"
    test_dir.mkdir()

    result = check_checkpoint_valid(test_dir)
    assert result is False


def test_check_checkpoint_valid_only_aside_sha_sums(tmp_path):
    """Test check_checkpoint_valid with only aside sha_sums file."""
    test_dir = tmp_path / "test_checkpoint"
    test_dir.mkdir()

    # Create only the aside sha_sums file
    aside_sha_sums_file = tmp_path / "sha256_sums.txt"
    aside_sha_sums_file.write_text("test_content")

    result = check_checkpoint_valid(test_dir)
    assert result is False


def test_check_checkpoint_valid_only_inside_sha_sums(tmp_path):
    """Test check_checkpoint_valid with only inside sha_sums file."""
    test_dir = tmp_path / "test_checkpoint"
    test_dir.mkdir()

    # Create only the inside sha_sums file
    inside_sha_sums_file = test_dir / "sha256_sums.txt"
    inside_sha_sums_file.write_text("test_content")

    result = check_checkpoint_valid(test_dir)
    assert result is False


@pytest.mark.parametrize("sha_type", ["256", "512"])
def test_check_checkpoint_valid_matching_sha_sums_files(tmp_path, sha_type):
    """Test check_checkpoint_valid with matching sha_sums files."""
    test_dir = tmp_path / "test_checkpoint"
    test_dir.mkdir()

    # Create both sha files with matching content
    aside_sha_file = tmp_path / f"sha{sha_type}_sums.txt"
    inside_sha_file = test_dir / f"sha{sha_type}_sums.txt"
    test_content = "test_content"

    aside_sha_file.write_text(test_content)
    inside_sha_file.write_text(test_content)

    result = check_checkpoint_valid(test_dir)
    assert result is True


@pytest.mark.parametrize("sha_type", ["256", "512"])
def test_check_checkpoint_valid_different_sha_sums_files(tmp_path, sha_type):
    """Test check_checkpoint_valid with different sha_sums files."""
    test_dir = tmp_path / "test_checkpoint"
    test_dir.mkdir()

    # Create both sha files with different content
    aside_sha_file = tmp_path / f"sha{sha_type}_sums.txt"
    inside_sha_file = test_dir / f"sha{sha_type}_sums.txt"

    aside_sha_file.write_text("content1")
    inside_sha_file.write_text("content2")

    result = check_checkpoint_valid(test_dir)
    assert result is False


@pytest.mark.parametrize("sha_type", ["256", "512"])
def test_get_sha_sums_path(sha_type):
    """Test get_sha_sums_path function."""
    from pathlib import Path

    path = Path("/test/path")

    # Test SHA-256 (default)
    sha_path = get_sha_sums_path(path, sha_type)
    assert sha_path == path / (path.stem + f"_sha{sha_type}_sums.txt")
