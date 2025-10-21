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
"""Toy backends for unit tests."""

import time
from pathlib import Path
from typing import Any

from aitune.torch.backend.backend import Backend


class SleepBackend(Backend):
    """Backend that simulate an inference using sleep()."""

    def __init__(self, sleep_time: float = 0.001):
        super().__init__()
        self.sleep_time = sleep_time

    def key(self) -> str:
        return f"{self.__class__.__name__}"

    def describe(self) -> str:
        return "SleepBackend - test only"

    def _build(self, module: Any, graph_spec: Any, data: list[Any], cache_dir: Path) -> Backend:
        self._activate()
        self._module = module
        return self

    def _activate(self):
        pass

    def _infer(self, *args, **kwargs):
        time.sleep(self.sleep_time)
        return self._module(*args, **kwargs)

    def _deactivate(self):
        pass

    def _deploy(self):
        pass

    def is_jit(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__} with sleep time {self.sleep_time}"

    def to_dict(self):
        return {"sleep_time": self.sleep_time}

    @classmethod
    def from_dict(cls, state_dict: dict):
        return cls(sleep_time=state_dict["sleep_time"])


class BuildFailsBackend(SleepBackend):
    """Backend that fails on build with a given exception.

    Args:
        exception_class: The exception class to raise on build.
    """

    def __init__(self, exception_class: type[Exception]):
        super().__init__()
        self.exception_class = exception_class

    def _build(self, module: Any, graph_spec: Any, data: list[Any], cache_dir: Path) -> Backend:
        raise self.exception_class("Build failed")
