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
"""Storage tasks."""

import hashlib
import logging
import shutil
import zipfile
from abc import ABC, abstractmethod
from collections import deque
from itertools import count
from pathlib import Path, PosixPath
from typing import Literal

import torch

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
    def load(self, path: Path) -> dict:
        """Load a state dictionary from the specified path.

        Args:
            path: The path from where the state dictionary should be loaded.

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

    def load(self, path: Path) -> dict:
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


class CopyBackendArtifactsTask(SaveTask):
    """Task to copy backend artifacts."""

    def save(self, target_path: Path, state_dict: dict) -> None:
        """Copy backend artifacts to the specified path.

        Function traverses through state_dict which can contain torch parameters, layers or AITune backend artifacts.
        If it finds a backend artifact, it looks for any Path and copies it to the specified target path.

        It enumerates (with a counter) copied artifacts to avoid name collisions.

        Args:
            target_path: The path where the backend artifacts should be copied.
            state_dict: The state dictionary to traverse.
        """
        dicts_to_check = deque([state_dict])
        counter = count(1)

        while dicts_to_check:
            current_dict = dicts_to_check.pop()
            for property_name, value in current_dict.items():
                if isinstance(value, dict):
                    # allow traversal through nested dictionaries which can contain backend artifacts
                    dicts_to_check.append(value)
                elif property_name == TunedModule.BACKENDS_KEY:
                    # value is a backends_data (see TunedModule.to_dict())
                    for _, backend_data in value:
                        for backend_property_name, backend_value in backend_data.items():
                            # look for Path objects
                            if isinstance(backend_value, Path):
                                new_path = target_path / f"{next(counter)}_{backend_value.name}"
                                shutil.copy(backend_value, new_path)
                                backend_data[backend_property_name] = new_path  # overwrite with new path


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
                sha_hashes.append((str(file_path), sha_hash))

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


class ShaSumsLoadTask(LoadTask):
    """Task to load and verify SHA hashes of files."""

    def __init__(self, sha_type: Literal["256", "512"] = "256"):
        """Initialize the task.

        Args:
            sha_type: The SHA type to use for hashing. Must be either "256" or "512".
        """
        self.sha_type = sha_type

    def load(self, path: Path) -> dict:
        """Load and verify SHA hashes of files in the specified path.

        Args:
            path: The path containing files to verify.

        Raises:
            FileNotFoundError: If sha_hashes.txt file is not found.
            ValueError: If sha_hashes.txt file format is invalid.
        """
        sha_file_path = get_sha_sums_path(path, self.sha_type)

        if not sha_file_path.exists():
            raise FileNotFoundError(f"SHA hash file not found: {sha_file_path}")

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

        failed_files = []
        for file_path, hash_value in stored_hashes.items():
            current_hash = calculate_file_sha_hash(file_path, self.sha_type)
            if current_hash != hash_value:
                failed_files.append(file_path)

        if failed_files:
            raise ValueError(f"Failed to verify SHA hashes for files: {failed_files}")

        return {}


class UnzipLoadTask(LoadTask):
    """Task to load a state dictionary from a zip file."""

    def load(self, path: Path) -> dict:
        """Extract the zip file at the given path to a folder with the same name.

        Args:
            path: The path to the zip file that should be extracted.

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
    """Doest not use safe globals to load torch checkpoint with PosixPath object allowed."""
    with torch.serialization.safe_globals([PosixPath]):
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
