# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Storage tasks."""

import hashlib
import logging
import shutil
import zipfile
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from functools import partial
from itertools import count
from pathlib import Path, PosixPath
from typing import Any, Literal

import torch

from aitune.torch.checkpoint.artifact import ArtifactPath
from aitune.torch.module.tuned_module import TunedModule
from aitune.torch.utils.path_utils import format_file_size, get_file_size

AIT_EXTENSION = ".ait"
STATE_DICT_FILE = "state_dict.pt"

logger = logging.getLogger(__name__)


class SaveTask(ABC):
    """Base class to save state dict."""

    @abstractmethod
    def save(self, path: Path, state_dict: dict) -> None:
        """Save the state dictionary to the specified path.

        Args:
            path: The path where the state dictionary should be saved.
            state_dict: The state dictionary to save.
        """
        raise NotImplementedError("Subclass must implement this method")


class LoadTask(ABC):
    """Base class to load state dict."""

    @abstractmethod
    def load(self, path: Path, state_dict: dict | None = None) -> dict:
        """Load a state dictionary from the specified path.

        Args:
            path: The path from where the state dictionary should be loaded.
            state_dict: Accumulated state dictionary loaded by previous tasks.

        Returns:
            dict: The loaded state dictionary.
        """
        raise NotImplementedError("Subclass must implement this method")


class TorchSaveTask(SaveTask):
    """Task to save a state dictionary using torch.save."""

    def save(self, path: Path, state_dict: dict) -> None:
        """Save the state dictionary to the specified path.

        Args:
            path: The path where the state dictionary should be saved.
            state_dict: The state dictionary to save.
        """
        file_path = path / STATE_DICT_FILE
        torch.save(state_dict, file_path)

        file_size = get_file_size(file_path)
        logger.info("✅ State dict saved to %s [%s]", file_path, format_file_size(file_size))


class TorchLoadTask(LoadTask):
    """Task to load a state dictionary using torch.load."""

    def load(self, path: Path, state_dict: dict | None = None) -> dict:
        """Load a state dictionary from the specified path."""
        file_path = path / STATE_DICT_FILE
        if not file_path.exists():
            raise FileNotFoundError(f"State dictionary file not found: {file_path}")

        state_dict = torch_load_with_custom_types(file_path)

        file_size = get_file_size(file_path)
        logger.info("✅ State dict loaded from %s [%s]", file_path, format_file_size(file_size))

        return state_dict


class MakeFolderTask(SaveTask):
    """Task to make a folder."""

    def __init__(self, overwrite: bool = True):
        """Initialize the task.

        Args:
            overwrite: Whether to overwrite the folder if it already exists.
        """
        self.overwrite = overwrite

    def save(self, path: Path, state_dict: dict) -> None:
        """Make a folder."""
        if path.exists():
            if self.overwrite:
                shutil.rmtree(path)
            else:
                raise FileExistsError(f"Folder {path} already exists")
        path.mkdir(parents=True, exist_ok=True)


class RemoveFolderTask(SaveTask):
    """Task to remove a folder."""

    def save(self, path: Path, state_dict: dict) -> None:
        """Remove a folder."""
        if path.exists():
            shutil.rmtree(path)


def _iter_backend_data_dicts(state_dict: dict):
    """Yield backend state dictionaries that may contain checkpoint artifacts."""
    dicts_to_check = deque([state_dict])

    while dicts_to_check:
        current_dict = dicts_to_check.pop()
        for property_name, value in current_dict.items():
            if isinstance(value, dict):
                dicts_to_check.append(value)
            elif property_name == TunedModule.BACKENDS_KEY:
                for _, backend_data in value:
                    yield backend_data


def _transform_artifact_paths(value: Any, transform: Callable[[ArtifactPath], ArtifactPath]) -> Any:
    """Transform artifact paths in plain checkpoint collections in place where possible.

    Collection subclasses may encode user-visible behavior, such as positional indexing in
    ``transformers.ModelOutput``. Treat them as opaque values so checkpoint processing does not
    replace them with their plain built-in counterparts. Tuples are rebuilt because they are
    immutable.
    """
    if isinstance(value, ArtifactPath):
        return transform(value)
    if type(value) is dict:
        for key, item in value.items():
            value[key] = _transform_artifact_paths(item, transform)
        return value
    if type(value) is list:
        for index, item in enumerate(value):
            value[index] = _transform_artifact_paths(item, transform)
        return value
    if type(value) is tuple:
        return tuple(_transform_artifact_paths(item, transform) for item in value)
    return value


class CopyBackendArtifactsTask(SaveTask):
    """Task to copy backend artifacts."""

    def save(self, target_path: Path, state_dict: dict) -> None:
        """Copy backend artifacts to the specified path.

        Each backend and each distinct artifact root receives its own numbered
        directory. Paths relative to an artifact root are preserved.

        Args:
            target_path: The path where the backend artifacts should be copied.
            state_dict: The state dictionary to traverse.
        """
        counter = count(1)

        for backend_data in _iter_backend_data_dicts(state_dict):
            backend_dir = target_path / str(next(counter))
            backend_dir.mkdir()
            artifact_root_dirs: dict[Path, Path] = {}
            copy_artifact = partial(
                self._copy_artifact,
                target_path=target_path,
                backend_dir=backend_dir,
                artifact_root_dirs=artifact_root_dirs,
            )
            _transform_artifact_paths(backend_data, copy_artifact)

    def _copy_artifact(
        self,
        artifact: ArtifactPath,
        *,
        target_path: Path,
        backend_dir: Path,
        artifact_root_dirs: dict[Path, Path],
    ) -> ArtifactPath:
        """Copy an artifact into the checkpoint while preserving its root-relative path.

        The destination path has the following structure::

            <target_path>/<backend index>/<artifact root index>/<artifact.relative_path>

        ``backend_dir`` already contains the backend index. Each distinct resolved
        ``artifact.root`` is assigned a numbered directory below ``backend_dir``;
        artifacts sharing a root also share that directory. The artifact's relative
        path is appended unchanged, preserving its original directory structure.

        Args:
            artifact: Artifact to copy, including its owning root and relative path.
            target_path: Root directory of the checkpoint being created.
            backend_dir: Numbered checkpoint directory assigned to the artifact's backend.
            artifact_root_dirs: Mapping from resolved artifact roots to their numbered
                directories within ``backend_dir``.

        Returns:
            An artifact pointing to the copied checkpoint entry. Its root is ``.``
            and its relative path is relative to ``target_path``, so it can be
            rebased when the checkpoint is loaded from another location.

        Raises:
            ValueError: If the resolved artifact path is outside its declared root.
        """
        source_root = artifact.root.resolve()
        source_path = artifact.path.resolve()
        try:
            source_relative_path = source_path.relative_to(source_root)
        except ValueError as e:
            raise ValueError(f"Backend artifact path resolves outside its root: {artifact.path}") from e

        artifact_root_dir = artifact_root_dirs.get(source_root)
        if artifact_root_dir is None:
            # create a new artifact root directory
            artifact_root_dir = backend_dir / str(len(artifact_root_dirs) + 1)
            artifact_root_dirs[source_root] = artifact_root_dir

        copied_path = artifact_root_dir / source_relative_path
        if source_path.is_dir():
            shutil.copytree(source_path, copied_path, dirs_exist_ok=True)
        elif not copied_path.exists():
            copied_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source_path, copied_path)

        return ArtifactPath(root=Path("."), relative_path=copied_path.relative_to(target_path))


class ShaSumsSaveTask(SaveTask):
    """Task to save SHA hashes of files in the state dictionary."""

    def __init__(self, sha_type: Literal["256", "512"] = "256"):
        """Initialize the task.

        Args:
            sha_type: The SHA type to use for hashing. Must be either "256" or "512".
        """
        self.sha_type = sha_type

    def save(self, path: Path, state_dict: dict) -> None:
        """Save SHA hashes of files in the specified path to a file.

        Args:
            path: The path containing files to hash.
            state_dict: The state dictionary (not used in this implementation).
        """
        sha_hashes = []

        # Iterate over all files in the specified path
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file():
                # Calculate SHA hash of the file
                sha_hash = calculate_file_sha_hash(file_path, self.sha_type)
                sha_hashes.append((file_path.relative_to(path).as_posix(), sha_hash))

        # Write SHA hashes to file
        sha_file_path = get_sha_sums_path(path, self.sha_type)
        with sha_file_path.open("w", encoding="utf-8") as fp:
            for file_path, sha_hash in sha_hashes:
                fp.write(f"{sha_hash}  {file_path}\n")


class ZipSaveTask(SaveTask):
    """Task to save a state dictionary to a zip file."""

    def save(self, path: Path, state_dict: dict) -> None:
        """Compress the folder under the given path to a zip file, including all nested folders.

        Args:
            path: The path to the folder that should be compressed to a zip file.
            state_dict: The state dictionary (not used in this implementation).

        Raises:
            FileNotFoundError: If the folder does not exist.
            ValueError: If the path is not a directory.
        """
        zip_path = path.with_name(path.name + AIT_EXTENSION)

        logger.info("🔄 Compressing checkpoint...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zip_fp:
            # Recursively find all files in the directory tree
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    # Calculate relative path from the root folder to preserve directory structure
                    file_path_in_zip = file_path.relative_to(path)
                    zip_fp.write(file_path, file_path_in_zip)

        zip_file_size = get_file_size(zip_path)
        logger.info("✅ Checkpoint compressed and saved to %s [%s]", zip_path, format_file_size(zip_file_size))
        # copy sha hashes aside zip if they exist
        # Try both SHA-256 and SHA-512 files
        for sha_type in ["256", "512"]:
            sha_file_path = get_sha_sums_path(path, sha_type)
            if sha_file_path.exists():
                copy_sha_file_path = path.parent / sha_file_path.name
                shutil.copy(sha_file_path, copy_sha_file_path)
                logger.info("✅ SHA hash file copied to %s", copy_sha_file_path)
                break


class RelocateBackendArtifactsTask(LoadTask):
    """Rebase saved backend artifact paths onto the extracted checkpoint.

    New checkpoints identify artifacts with :class:`ArtifactPath`. Checkpoints
    created before ``ArtifactPath`` used direct, checkpoint-relative ``Path``
    values in backend state dictionaries; those values still load, but the format
    is deprecated and a warning is emitted.
    """

    def load(self, path: Path, state_dict: dict | None = None) -> dict:
        """Resolve relative backend artifact paths and return no additional state updates."""
        if state_dict is None:
            raise ValueError("RelocateBackendArtifactsTask requires loaded state_dict")

        checkpoint_root = path.resolve()
        relocate_artifact = partial(self._relocate_artifact, checkpoint_root=checkpoint_root)
        used_legacy_format = False
        for backend_index, backend_data in enumerate(_iter_backend_data_dicts(state_dict), start=1):
            _transform_artifact_paths(backend_data, relocate_artifact)
            if self._relocate_legacy_paths(checkpoint_root, backend_index, backend_data):
                used_legacy_format = True

        if used_legacy_format:
            logger.warning(
                "⚠️ This checkpoint uses a deprecated format. It still loads, "
                "but support will be removed in a future release."
            )

        return {}

    @staticmethod
    def _relocate_artifact(artifact: ArtifactPath, *, checkpoint_root: Path) -> ArtifactPath:
        """Rebase a checkpoint-relative artifact onto the extracted checkpoint root.

        Saved artifacts contain a path relative to the checkpoint directory, for
        example ``1/1/nested/model.bin``. This method resolves that entry below
        ``checkpoint_root`` and returns an artifact whose root is the absolute
        checkpoint directory while preserving the checkpoint-relative path.

        Args:
            artifact: Artifact path restored from the checkpoint state dictionary.
            checkpoint_root: Resolved root directory of the extracted checkpoint.

        Returns:
            An artifact rebased onto ``checkpoint_root``.

        Raises:
            ValueError: If the resolved artifact path is outside ``checkpoint_root``.
            FileNotFoundError: If the referenced artifact does not exist in the checkpoint.
        """
        checkpoint_relative_path = artifact.path
        artifact_path = (checkpoint_root / checkpoint_relative_path).resolve()
        try:
            artifact_path.relative_to(checkpoint_root)
        except ValueError as e:
            raise ValueError(f"Backend artifact path resolves outside checkpoint: {checkpoint_relative_path}") from e
        if not artifact_path.exists():
            raise FileNotFoundError(f"Backend artifact not found: {artifact_path}")
        return ArtifactPath(root=checkpoint_root, relative_path=checkpoint_relative_path)

    @staticmethod
    def _relocate_legacy_paths(checkpoint_root: Path, backend_index: int, backend_data: dict) -> bool:
        """Convert top-level artifact paths written by the legacy checkpoint format.

        Before ``ArtifactPath`` was introduced, copied backend artifacts were
        stored directly as checkpoint-relative ``Path`` values. Their first path
        component was the numbered backend directory, for example
        ``Path("1/model.plan")`` for the first backend. A relative ``Path`` is
        treated as a legacy artifact only when this component matches
        ``backend_index``; other relative paths may be regular backend configuration
        and are left unchanged.

        A recognized path is resolved below ``checkpoint_root``, checked for path
        traversal and existence, and replaced in ``backend_data`` with an
        ``ArtifactPath`` rooted at the extracted checkpoint. Absolute ``Path``
        values are left unchanged. Only values directly in ``backend_data`` are
        considered because built-in backends stored legacy artifacts at that level.

        Args:
            checkpoint_root: Resolved root directory of the extracted checkpoint.
            backend_index: One-based index identifying the backend's legacy artifact directory.
            backend_data: Backend state dictionary to update in place.

        Returns:
            True if at least one legacy path was converted.

        Raises:
            ValueError: If a recognized legacy path resolves outside ``checkpoint_root``.
            FileNotFoundError: If the referenced legacy artifact does not exist.
        """
        converted = False
        for key, value in backend_data.items():
            if not isinstance(value, Path) or value.is_absolute():
                continue
            if not value.parts or value.parts[0] != str(backend_index):
                continue

            artifact_path = (checkpoint_root / value).resolve()
            try:
                artifact_path.relative_to(checkpoint_root)
            except ValueError as e:
                raise ValueError(f"Backend artifact path resolves outside checkpoint: {value}") from e
            if not artifact_path.exists():
                raise FileNotFoundError(f"Backend artifact not found: {artifact_path}")
            backend_data[key] = ArtifactPath(root=checkpoint_root, relative_path=value)
            converted = True
        return converted


class ShaSumsLoadTask(LoadTask):
    """Task to load and verify SHA hashes of files."""

    def __init__(self, sha_type: Literal["256", "512"] = "256"):
        """Initialize the task.

        Args:
            sha_type: The SHA type to use for hashing. Must be either "256" or "512".
        """
        self.sha_type = sha_type

    def load(self, path: Path, state_dict: dict | None = None) -> dict:
        """Load and verify SHA hashes of files in the specified path.

        Args:
            path: The path containing files to verify.
            state_dict: Unused accumulated state from prior load tasks.

        Raises:
            FileNotFoundError: If sha_hashes.txt file is not found.
            ValueError: If sha_hashes.txt file format is invalid.
        """
        sha_file_path = get_sha_sums_path(path, self.sha_type)

        if not sha_file_path.exists():
            candidates = sorted(path.glob(f"*_sha{self.sha_type}_sums.txt"))
            if len(candidates) == 1:
                sha_file_path = candidates[0]
            elif len(candidates) > 1:
                raise ValueError(f"Ambiguous SHA hash files for checkpoint {path}: {candidates}")
            else:
                raise FileNotFoundError(f"SHA hash file not found: {sha_file_path}")

        stored_hashes = self._load_stored_hashes(sha_file_path)
        failed_files = self._get_failed_files(path, stored_hashes)

        if failed_files:
            raise ValueError(f"Failed to verify SHA hashes for files: {failed_files}")

        return {}

    def _load_stored_hashes(self, sha_file_path: Path) -> dict[str, str]:
        """Load stored hash values from a SHA sums file."""
        stored_hashes = {}
        try:
            with open(sha_file_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split("  ", 1)  # Split on double space
                        if len(parts) != 2:
                            raise ValueError(f"Invalid hash file format in line: {line}")
                        hash_value, file_path = parts
                        stored_hashes[file_path] = hash_value
        except Exception as e:
            raise ValueError("Error reading SHA hash file") from e
        return stored_hashes

    def _get_failed_files(self, path: Path, stored_hashes: dict[str, str]) -> list[str]:
        """Return stored paths whose current file hashes do not match."""
        failed_files = []
        checkpoint_root = path.resolve()
        for stored_path_str, hash_value in stored_hashes.items():
            file_path = Path(stored_path_str)
            if not file_path.is_absolute():
                file_path = (checkpoint_root / file_path).resolve()
                try:
                    file_path.relative_to(checkpoint_root)
                except ValueError as e:
                    raise ValueError(f"Checkpoint file path resolves outside checkpoint: {stored_path_str}") from e
            current_hash = calculate_file_sha_hash(file_path, self.sha_type)
            if current_hash != hash_value:
                failed_files.append(stored_path_str)
        return failed_files


class UnzipLoadTask(LoadTask):
    """Task to load a state dictionary from a zip file."""

    def load(self, path: Path, state_dict: dict | None = None) -> dict:
        """Extract the zip file at the given path to a folder with the same name.

        Args:
            path: The path to the zip file that should be extracted.
            state_dict: Unused accumulated state from prior load tasks.

        Returns:
            dict: An empty dictionary (following the pattern of other load tasks).

        Raises:
            FileNotFoundError: If the zip file does not exist.
            ValueError: If the path is not a zip file.
        """
        zip_path = path.with_name(path.name + AIT_EXTENSION)
        if not zip_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {zip_path}")

        file_size = get_file_size(zip_path)
        logger.info("🔄 Extracting checkpoint from: %s [%s]", path, format_file_size(file_size))

        if not check_checkpoint_valid(path):
            with zipfile.ZipFile(zip_path, "r") as zip_fp:
                zip_fp.extractall(path)

        logger.info("✅ Checkpoint extracted")

        return {}


def torch_load_with_custom_types(path: Path) -> dict:
    """Load a checkpoint safely with AITune's serialized value types allowlisted."""
    from aitune.torch.dynamic_shapes import BatchDim, DynamicDim
    from aitune.torch.module.locator import Locator, ObjectType
    from aitune.torch.module.sample_metadata import SampleMetadata
    from aitune.torch.module.tensor_spec import TensorSpec

    safe_globals = [ArtifactPath, BatchDim, DynamicDim, Locator, ObjectType, PosixPath, SampleMetadata, TensorSpec]
    with torch.serialization.safe_globals(safe_globals):
        # unfortunately, we cannot use weights_only=True here because some backends store internal objects
        # until this is not fixed, we need weights_only=False
        return torch.load(path, weights_only=False)


def calculate_file_sha_hash(file_path: str | Path, sha_type: Literal["256", "512"] = "256") -> str:
    """Calculate SHA hash of a file.

    Args:
        file_path: Path to the file to hash.
        sha_type: The SHA type to use for hashing. Must be either "256" or "512".

    Returns:
        str: SHA hash of the file as a hexadecimal string.
    """
    if sha_type == "256":
        sha_hash = hashlib.sha256()
    elif sha_type == "512":
        sha_hash = hashlib.sha512()
    else:
        raise ValueError("sha_type must be either '256' or '512'")

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha_hash.update(chunk)
    return sha_hash.hexdigest()


def get_sha_sums_path(path: Path, sha_type: Literal["256", "512"] = "256") -> Path:
    """Get the path to the SHA sums file.

    Args:
        path: The path to the checkpoint directory.
        sha_type: The SHA type used for hashing. Must be either "256" or "512".

    Returns:
        Path: The path to the SHA sums file.
    """
    hashes_filename = f"sha{sha_type}_sums.txt"
    return path / (path.stem + "_" + hashes_filename)


def check_checkpoint_valid(path: Path) -> bool:
    """Check if the checkpoint is valid.

    Args:
        path: The path to the checkpoint.

    Returns:
        bool: True if the checkpoint is valid, False otherwise.
        Checkpoint is valid if there is sha file aside path and inside path are the same.
    """
    if not path.exists():
        return False  # there is no unzipped folder

    # Try both SHA-256 and SHA-512 files
    for sha_type in ["256", "512"]:
        aside_sha_path = path.parent / f"sha{sha_type}_sums.txt"
        inside_sha_path = path / f"sha{sha_type}_sums.txt"

        if aside_sha_path.exists() and inside_sha_path.exists():
            return aside_sha_path.read_text(encoding="utf-8") == inside_sha_path.read_text(encoding="utf-8")

    return False
