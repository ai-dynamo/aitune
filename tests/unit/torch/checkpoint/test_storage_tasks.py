# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for model state storage functionality."""

import logging
import shutil
from pathlib import Path

import pytest

from aitune.torch.backend import ArtifactPath
from aitune.torch.checkpoint.storage_tasks import (
    AIT_EXTENSION,
    CopyBackendArtifactsTask,
    MakeFolderTask,
    RelocateBackendArtifactsTask,
    ShaSumsLoadTask,
    ShaSumsSaveTask,
    TorchLoadTask,
    TorchSaveTask,
    UnzipLoadTask,
    ZipSaveTask,
    calculate_file_sha_hash,
    check_checkpoint_valid,
    get_sha_sums_path,
)
from aitune.torch.module.tuned_module import TunedModule
from tests.toy_models.torch_models import ToyTorchModel
from tests.unit.torch.checkpoint.helpers import _backend_state_with_paths, _only_backend_data

_STORAGE_TASKS_LOGGER = "aitune.torch.checkpoint.storage_tasks"
_DEPRECATED_FORMAT_WARNING = "This checkpoint uses a deprecated format"


class _UnsafeCheckpointValue:
    pass


class _PositionalOutput(dict):
    """Dictionary-like output that also supports positional access."""

    def __getitem__(self, key):
        if isinstance(key, int):
            return tuple(self.values())[key]
        return super().__getitem__(key)


def test_save_load_torch_task(tmp_path):
    model = ToyTorchModel(is_linear=True)
    save_task = TorchSaveTask()
    load_task = TorchLoadTask()

    orig_state_dict = model.state_dict()
    orig_state_dict["extra_data"] = "for testing"
    orig_state_dict["artifact"] = ArtifactPath(tmp_path, "artifact.bin")

    save_task.save(tmp_path, orig_state_dict)
    state_dict = load_task.load(tmp_path)

    assert state_dict.keys() == orig_state_dict.keys()
    assert state_dict["extra_data"] == orig_state_dict["extra_data"]
    assert state_dict["artifact"] == orig_state_dict["artifact"]


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


def test_copy_backend_artifacts_preserves_root_relative_file_path(tmp_path):
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    artifact_root = tmp_path / "cache"
    artifact_path = artifact_root / "nested" / "source.plan"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"engine")
    state_dict = _backend_state_with_paths(
        engine_path=ArtifactPath(root=artifact_root, relative_path=Path("nested/source.plan"))
    )

    CopyBackendArtifactsTask().save(checkpoint_path, state_dict)

    backend_data = _only_backend_data(state_dict)
    assert backend_data["engine_path"] == ArtifactPath(root=Path("."), relative_path=Path("1/1/nested/source.plan"))
    assert (checkpoint_path / "1" / "1" / "nested" / "source.plan").read_bytes() == b"engine"


def test_copy_backend_artifacts_stores_relative_directory_paths(tmp_path):
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    artifact_dir = tmp_path / "compiled_model"
    artifact_dir.mkdir()
    (artifact_dir / "model.bin").write_bytes(b"compiled")
    state_dict = _backend_state_with_paths(compiled_model_path=ArtifactPath(root=artifact_dir, relative_path=Path(".")))

    CopyBackendArtifactsTask().save(checkpoint_path, state_dict)

    backend_data = _only_backend_data(state_dict)
    assert backend_data["compiled_model_path"] == ArtifactPath(root=Path("."), relative_path=Path("1/1"))
    assert (checkpoint_path / "1" / "1" / "model.bin").read_bytes() == b"compiled"


def test_copy_backend_artifacts_recurses_through_dicts_lists_and_tuples(tmp_path):
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    first_artifact = artifact_root / "first.plan"
    second_artifact = artifact_root / "metadata" / "second.json"
    second_artifact.parent.mkdir()
    first_artifact.write_bytes(b"engine")
    second_artifact.write_text("{}", encoding="utf-8")
    state_dict = _backend_state_with_paths(
        kernel_plan={
            "replacements": [
                {
                    "artifacts": (
                        ArtifactPath.from_existing(first_artifact, root=artifact_root),
                        {"metadata": [ArtifactPath.from_existing(second_artifact, root=artifact_root)]},
                    ),
                },
            ],
        },
    )

    CopyBackendArtifactsTask().save(checkpoint_path, state_dict)

    artifacts = _only_backend_data(state_dict)["kernel_plan"]["replacements"][0]["artifacts"]
    assert artifacts == (
        ArtifactPath(root=Path("."), relative_path=Path("1/1/first.plan")),
        {"metadata": [ArtifactPath(root=Path("."), relative_path=Path("1/1/metadata/second.json"))]},
    )
    assert (checkpoint_path / artifacts[0].path).read_bytes() == b"engine"
    assert (checkpoint_path / artifacts[1]["metadata"][0].path).read_text(encoding="utf-8") == "{}"


def test_copy_backend_artifacts_preserves_custom_output_collection_type(tmp_path):
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    artifact_path = tmp_path / "model.plan"
    artifact_path.write_bytes(b"engine")
    output = _PositionalOutput(last_hidden_state="hidden", pooler_output="pooled")
    state_dict = _backend_state_with_paths(
        engine_path=ArtifactPath.from_existing(artifact_path, root=tmp_path),
        output_object=output,
    )

    CopyBackendArtifactsTask().save(checkpoint_path, state_dict)

    restored_output = _only_backend_data(state_dict)["output_object"]
    assert type(restored_output) is _PositionalOutput
    assert restored_output[0] == "hidden"


def test_copy_backend_artifacts_leaves_plain_paths_unchanged(tmp_path):
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    timing_cache = tmp_path / "timing.cache"
    timing_cache.write_bytes(b"timings")
    state_dict = _backend_state_with_paths(config={"timing_cache": timing_cache})

    CopyBackendArtifactsTask().save(checkpoint_path, state_dict)

    assert _only_backend_data(state_dict)["config"]["timing_cache"] == timing_cache
    assert list((checkpoint_path / "1").iterdir()) == []


def test_copy_backend_artifacts_separates_distinct_roots_with_same_relative_path(tmp_path):
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    (first_root / "model.bin").write_bytes(b"first")
    (second_root / "model.bin").write_bytes(b"second")
    state_dict = _backend_state_with_paths(
        artifacts=[
            ArtifactPath(first_root, Path("model.bin")),
            ArtifactPath(second_root, Path("model.bin")),
        ]
    )

    CopyBackendArtifactsTask().save(checkpoint_path, state_dict)

    artifacts = _only_backend_data(state_dict)["artifacts"]
    assert artifacts == [
        ArtifactPath(Path("."), Path("1/1/model.bin")),
        ArtifactPath(Path("."), Path("1/2/model.bin")),
    ]
    assert (checkpoint_path / artifacts[0].path).read_bytes() == b"first"
    assert (checkpoint_path / artifacts[1].path).read_bytes() == b"second"


def test_copy_backend_artifacts_reuses_repeated_artifact(tmp_path):
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    artifact_path = tmp_path / "model.bin"
    artifact_path.write_bytes(b"model")
    artifact = ArtifactPath.from_existing(artifact_path, root=tmp_path)
    state_dict = _backend_state_with_paths(artifacts=[artifact, artifact])

    CopyBackendArtifactsTask().save(checkpoint_path, state_dict)

    artifacts = _only_backend_data(state_dict)["artifacts"]
    assert artifacts[0] == artifacts[1]
    assert list((checkpoint_path / "1" / "1").iterdir()) == [checkpoint_path / artifacts[0].path]


def test_relocate_backend_artifacts_resolves_relative_paths(tmp_path, caplog):
    checkpoint_path = tmp_path / "checkpoint"
    artifact_path = checkpoint_path / "1" / "1" / "model.plan"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"engine")
    state_dict = _backend_state_with_paths(
        engine_path=ArtifactPath(root=Path("."), relative_path=Path("1/1/model.plan"))
    )

    with caplog.at_level(logging.WARNING, logger=_STORAGE_TASKS_LOGGER):
        update = RelocateBackendArtifactsTask().load(checkpoint_path, state_dict)

    assert update == {}
    backend_data = _only_backend_data(state_dict)
    assert backend_data["engine_path"] == ArtifactPath(
        root=checkpoint_path.resolve(), relative_path=Path("1/1/model.plan")
    )
    assert backend_data["engine_path"].path == artifact_path.resolve()
    assert _DEPRECATED_FORMAT_WARNING not in caplog.text


def test_relocate_backend_artifacts_recurses_through_dicts_lists_and_tuples(tmp_path):
    checkpoint_path = tmp_path / "checkpoint"
    first_artifact = checkpoint_path / "1" / "1" / "first.plan"
    second_artifact = checkpoint_path / "1" / "1" / "second.json"
    first_artifact.parent.mkdir(parents=True)
    first_artifact.write_bytes(b"engine")
    second_artifact.write_text("{}", encoding="utf-8")
    state_dict = _backend_state_with_paths(
        kernel_plan={
            "replacements": [
                {
                    "artifacts": (
                        ArtifactPath(Path("."), Path("1/1/first.plan")),
                        {"metadata": [ArtifactPath(Path("."), Path("1/1/second.json"))]},
                    ),
                },
            ],
        },
    )

    update = RelocateBackendArtifactsTask().load(checkpoint_path, state_dict)

    assert update == {}
    artifacts = _only_backend_data(state_dict)["kernel_plan"]["replacements"][0]["artifacts"]
    assert artifacts == (
        ArtifactPath(checkpoint_path.resolve(), Path("1/1/first.plan")),
        {"metadata": [ArtifactPath(checkpoint_path.resolve(), Path("1/1/second.json"))]},
    )


def test_relocate_backend_artifacts_rejects_relative_path_escape(tmp_path):
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    state_dict = _backend_state_with_paths(engine_path=ArtifactPath(root=Path(".."), relative_path=Path("model.plan")))

    with pytest.raises(ValueError, match="outside checkpoint"):
        RelocateBackendArtifactsTask().load(checkpoint_path, state_dict)


def test_relocate_backend_artifacts_requires_existing_relative_path(tmp_path):
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    state_dict = _backend_state_with_paths(
        engine_path=ArtifactPath(root=Path("."), relative_path=Path("1/1/missing.plan"))
    )

    with pytest.raises(FileNotFoundError, match="Backend artifact not found"):
        RelocateBackendArtifactsTask().load(checkpoint_path, state_dict)


def test_relocate_backend_artifacts_keeps_absolute_paths_when_checkpoint_has_same_basename(tmp_path):
    checkpoint_path = tmp_path / "checkpoint"
    artifact_path = checkpoint_path / "1" / "model.plan"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"engine")
    old_absolute_path = tmp_path / "old_build" / "model.plan"
    old_absolute_path.parent.mkdir()
    old_absolute_path.write_bytes(b"stale source")
    state_dict = _backend_state_with_paths(engine_path=old_absolute_path)

    update = RelocateBackendArtifactsTask().load(checkpoint_path, state_dict)

    assert update == {}
    backend_data = _only_backend_data(state_dict)
    assert backend_data["engine_path"] == old_absolute_path
    assert backend_data["engine_path"].read_bytes() == b"stale source"


def test_relocate_backend_artifacts_supports_legacy_relative_paths(tmp_path, caplog):
    checkpoint_path = tmp_path / "checkpoint"
    artifact_path = checkpoint_path / "1" / "model.plan"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"engine")
    state_dict = _backend_state_with_paths(engine_path=Path("1/model.plan"))

    with caplog.at_level(logging.WARNING, logger=_STORAGE_TASKS_LOGGER):
        RelocateBackendArtifactsTask().load(checkpoint_path, state_dict)

    assert _only_backend_data(state_dict)["engine_path"] == ArtifactPath(
        root=checkpoint_path.resolve(), relative_path=Path("1/model.plan")
    )
    warning_records = [record for record in caplog.records if _DEPRECATED_FORMAT_WARNING in record.message]
    assert len(warning_records) == 1
    assert warning_records[0].levelno == logging.WARNING


def test_relocate_legacy_checkpoint_format_warns_once_for_multiple_backends(tmp_path, caplog):
    checkpoint_path = tmp_path / "checkpoint"
    first_artifact = checkpoint_path / "1" / "model.plan"
    second_artifact = checkpoint_path / "2" / "model.plan"
    first_artifact.parent.mkdir(parents=True)
    second_artifact.parent.mkdir(parents=True)
    first_artifact.write_bytes(b"first")
    second_artifact.write_bytes(b"second")
    state_dict = {
        "wrapped": {
            TunedModule.BACKENDS_KEY: [
                ({"sample": "first"}, {TunedModule.TYPE_KEY: "FakeBackend", "engine_path": Path("1/model.plan")}),
                ({"sample": "second"}, {TunedModule.TYPE_KEY: "FakeBackend", "engine_path": Path("2/model.plan")}),
            ]
        }
    }

    with caplog.at_level(logging.WARNING, logger=_STORAGE_TASKS_LOGGER):
        RelocateBackendArtifactsTask().load(checkpoint_path, state_dict)

    warning_records = [record for record in caplog.records if _DEPRECATED_FORMAT_WARNING in record.message]
    assert len(warning_records) == 1


def test_relocate_backend_artifacts_rejects_legacy_path_escape(tmp_path):
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    state_dict = _backend_state_with_paths(engine_path=Path("1/../../model.plan"))

    with pytest.raises(ValueError, match="outside checkpoint"):
        RelocateBackendArtifactsTask().load(checkpoint_path, state_dict)


def test_relocate_backend_artifacts_requires_existing_legacy_path(tmp_path):
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    state_dict = _backend_state_with_paths(engine_path=Path("1/missing.plan"))

    with pytest.raises(FileNotFoundError, match="Backend artifact not found"):
        RelocateBackendArtifactsTask().load(checkpoint_path, state_dict)


def test_relocate_backend_artifacts_leaves_plain_relative_paths_unchanged(tmp_path):
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    timing_cache = Path("shared/timing.cache")
    state_dict = _backend_state_with_paths(config={"timing_cache": timing_cache})

    RelocateBackendArtifactsTask().load(checkpoint_path, state_dict)

    assert _only_backend_data(state_dict)["config"]["timing_cache"] == timing_cache


def test_relocate_backend_artifacts_requires_loaded_state_dict(tmp_path):
    with pytest.raises(ValueError, match="requires loaded state_dict"):
        RelocateBackendArtifactsTask().load(tmp_path, None)


def test_sha_sums_use_relative_paths_after_checkpoint_move(tmp_path):
    source_path = tmp_path / "source_checkpoint"
    source_path.mkdir()
    (source_path / "state_dict.pt").write_bytes(b"state")
    artifact_dir = source_path / "1"
    artifact_dir.mkdir()
    (artifact_dir / "model.plan").write_bytes(b"engine")

    ShaSumsSaveTask().save(source_path, {})
    sha_sums_file = get_sha_sums_path(source_path)
    sha_sums_text = sha_sums_file.read_text(encoding="utf-8")
    assert str(source_path) not in sha_sums_text
    assert "  state_dict.pt\n" in sha_sums_text
    assert "  1/model.plan\n" in sha_sums_text

    moved_path = tmp_path / "moved_checkpoint"
    shutil.copytree(source_path, moved_path)
    shutil.rmtree(source_path)

    assert ShaSumsLoadTask().load(moved_path) == {}


def test_sha_sums_load_uses_existing_absolute_entries_as_is(tmp_path):
    old_path = tmp_path / "old_checkpoint"
    old_path.mkdir()
    old_state_dict = old_path / "state_dict.pt"
    old_state_dict.write_bytes(b"state")
    old_hash = calculate_file_sha_hash(old_state_dict)

    moved_path = tmp_path / "moved_checkpoint"
    moved_path.mkdir()
    moved_state_dict = moved_path / "state_dict.pt"
    moved_state_dict.write_bytes(b"different checkpoint state")
    get_sha_sums_path(moved_path).write_text(f"{old_hash}  {old_state_dict}\n", encoding="utf-8")

    assert ShaSumsLoadTask().load(moved_path) == {}


def test_sha_sums_load_fails_missing_absolute_entries(tmp_path):
    old_path = tmp_path / "old_checkpoint"
    old_state_dict = old_path / "state_dict.pt"

    moved_path = tmp_path / "moved_checkpoint"
    moved_path.mkdir()
    get_sha_sums_path(moved_path).write_text(f"0  {old_state_dict}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        ShaSumsLoadTask().load(moved_path)


def test_sha_sums_load_fails_mismatched_absolute_entries(tmp_path):
    old_path = tmp_path / "old_checkpoint"
    old_path.mkdir()
    old_state_dict = old_path / "state_dict.pt"
    old_state_dict.write_bytes(b"state")
    old_hash = calculate_file_sha_hash(old_state_dict)
    old_state_dict.write_bytes(b"changed")

    moved_path = tmp_path / "moved_checkpoint"
    moved_path.mkdir()
    get_sha_sums_path(moved_path).write_text(f"{old_hash}  {old_state_dict}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Failed to verify SHA hashes"):
        ShaSumsLoadTask().load(moved_path)


def test_sha_sums_load_rejects_relative_path_escape(tmp_path):
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    outside_file = tmp_path / "outside.bin"
    outside_file.write_bytes(b"outside")
    outside_hash = calculate_file_sha_hash(outside_file)
    get_sha_sums_path(checkpoint_path).write_text(f"{outside_hash}  ../outside.bin\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside checkpoint"):
        ShaSumsLoadTask().load(checkpoint_path)


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
