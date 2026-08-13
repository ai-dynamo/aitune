# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for checkpoint artifact paths."""

from pathlib import Path

import pytest

from aitune.torch.backend import ArtifactPath as PublicArtifactPath
from aitune.torch.checkpoint.artifact import ArtifactPath


def test_artifact_path_is_exposed_as_backend_api():
    assert PublicArtifactPath is ArtifactPath


def test_artifact_path_joins_root_and_relative_path(tmp_path):
    artifact = ArtifactPath(tmp_path, "nested/model.bin")

    assert artifact.path == tmp_path / "nested" / "model.bin"


def test_artifact_path_string_is_effective_path(tmp_path):
    artifact = ArtifactPath(tmp_path, "nested/model.bin")

    assert str(artifact) == str(tmp_path / "nested" / "model.bin")


def test_artifact_path_normalizes_string_paths(tmp_path):
    artifact = ArtifactPath(root=str(tmp_path), relative_path="model.bin")

    assert artifact.root == tmp_path
    assert artifact.relative_path == Path("model.bin")


def test_artifact_path_from_existing(tmp_path):
    artifact_path = tmp_path / "nested" / "model.bin"
    artifact_path.parent.mkdir()
    artifact_path.touch()

    artifact = ArtifactPath.from_existing(artifact_path, root=tmp_path)

    assert artifact == ArtifactPath(tmp_path, "nested/model.bin")


def test_artifact_path_rejects_absolute_relative_path(tmp_path):
    with pytest.raises(ValueError, match="must be relative"):
        ArtifactPath(root=tmp_path, relative_path=tmp_path / "model.bin")


def test_artifact_path_rejects_root_escape(tmp_path):
    with pytest.raises(ValueError, match="must not escape"):
        ArtifactPath(root=tmp_path, relative_path=Path("../model.bin"))


def test_artifact_path_from_existing_rejects_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ArtifactPath.from_existing(tmp_path / "model.bin", root=tmp_path)


def test_artifact_path_from_existing_rejects_path_outside_root(tmp_path):
    artifact_path = tmp_path / "model.bin"
    artifact_path.touch()

    with pytest.raises(ValueError, match="outside root"):
        ArtifactPath.from_existing(artifact_path, root=tmp_path / "nested")
