# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint artifact path representation."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, init=False)
class ArtifactPath:
    """Identify a checkpoint artifact relative to its owning directory.

    ``root`` is the directory whose structure should be preserved and
    ``relative_path`` locates the artifact within it. The effective filesystem
    location is available through :attr:`path`.
    """

    root: Path
    relative_path: Path

    def __init__(self, root: str | Path, relative_path: str | Path) -> None:
        """Create an artifact whose path is relative to its owning root."""
        root = Path(root)
        relative_path = Path(relative_path)
        if relative_path.is_absolute():
            raise ValueError("Artifact relative_path must be relative")
        if ".." in relative_path.parts:
            raise ValueError("Artifact relative_path must not escape its root")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "relative_path", relative_path)

    @property
    def path(self) -> Path:
        """Return the effective filesystem path of the artifact."""
        return self.root / self.relative_path

    def __str__(self) -> str:
        """Return the effective filesystem path for display."""
        return str(self.path)

    @classmethod
    def from_existing(cls, path: str | Path, *, root: str | Path) -> "ArtifactPath":
        """Create an artifact from an existing path within an owning root."""
        path = Path(path)
        root = Path(root)
        if not path.exists():
            raise FileNotFoundError(f"Artifact path does not exist: {path}")
        try:
            relative_path = path.resolve().relative_to(root.resolve())
        except ValueError as e:
            raise ValueError(f"Artifact path {path} is outside root {root}") from e
        return cls(root=root, relative_path=relative_path)
