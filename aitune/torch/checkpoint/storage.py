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
"""Storage abstract base class."""

from pathlib import Path

from aitune.torch.checkpoint.storage_tasks import AIT_EXTENSION, LoadTask, SaveTask


class Storage:
    """Base class to save/load state dict."""

    def __init__(self, save_tasks: list[SaveTask], load_tasks: list[LoadTask]):
        """Initialize the storage.

        Args:
            save_tasks: The tasks to perform when saving.
            load_tasks: The tasks to perform when loading.
        """
        self.save_tasks = list(filter(None, save_tasks))
        self.load_tasks = list(filter(None, load_tasks))

    def save(self, path: str | Path, state_dict: dict) -> None:
        """Save the state dictionary to the specified path.

        Args:
            path: The path where the state dictionary should be saved - either a directory or a checkpoint file.
            state_dict: The state dictionary to save.
        """
        path = self._get_target_folder_path(path)

        for task in self.save_tasks:
            task.save(path, state_dict)

    def load(self, path: str | Path) -> dict:
        """Load a state dictionary from the specified path.

        Args:
            path: The path from where the state dictionary should be loaded- either a directory or a checkpoint file.

        Returns:
            dict: The loaded state dictionary.
        """
        path = self._get_target_folder_path(path)

        state_dict = {}
        for task in self.load_tasks:
            state_dict.update(task.load(path))
        return state_dict

    def _get_target_folder_path(self, path: str | Path) -> Path:
        """Get the target folder path.

        Args:
            path: The folder or a checkpoint file.

        Returns:
            The target folder path.
        """
        if not isinstance(path, Path):
            path = Path(path)

        if path.suffix == AIT_EXTENSION:
            path = path.with_suffix("")
        return path
