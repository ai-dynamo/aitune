# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for checkpoint storage orchestration."""

from pathlib import Path

from aitune.torch.checkpoint.storage import Storage
from aitune.torch.checkpoint.storage_tasks import LoadTask


class RecordingLoadTask(LoadTask):
    """Load task that records the accumulated state it received."""

    def __init__(self, update: dict):
        self.update = update
        self.seen_path = None
        self.seen_state = None

    def load(self, path: Path, state_dict: dict | None = None) -> dict:
        self.seen_path = path
        self.seen_state = dict(state_dict or {})
        return self.update


def test_storage_load_passes_accumulated_state_to_load_tasks(tmp_path):
    first = RecordingLoadTask({"a": 1})
    second = RecordingLoadTask({"b": 2})
    storage = Storage(save_tasks=[], load_tasks=[first, second])

    loaded = storage.load(tmp_path / "checkpoint")

    assert loaded == {"a": 1, "b": 2}
    assert first.seen_path == tmp_path / "checkpoint"
    assert first.seen_state == {}
    assert second.seen_path == tmp_path / "checkpoint"
    assert second.seen_state == {"a": 1}


def test_storage_load_merges_task_updates_before_next_task(tmp_path):
    first = RecordingLoadTask({"shared": "first", "first_only": True})
    second = RecordingLoadTask({"shared": "second"})
    storage = Storage(save_tasks=[], load_tasks=[first, second])

    loaded = storage.load(tmp_path / "checkpoint.ait")

    assert loaded == {"shared": "second", "first_only": True}
    assert second.seen_path == tmp_path / "checkpoint"
    assert second.seen_state == {"shared": "first", "first_only": True}
