# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ParentReceiver."""

import time
from collections.abc import Generator
from pathlib import Path
from queue import Queue

import pandas as pd
import pytest

from aitune.utils.monitoring.hardware_metrics.parent_receiver import ParentReceiver


@pytest.fixture
def receiver_and_queue() -> Generator[tuple[ParentReceiver, Queue], None, None]:
    """Fixture to create a receiver and queue."""
    queue = Queue()
    receiver = ParentReceiver(queue)
    try:
        yield receiver, queue
    finally:
        if receiver.ipc is not None:
            receiver.stop()


def test_parent_receiver_snapshot_sends_correct_command(tmp_path):
    """snapshot() sends the correct dict command to the child IPC and waits for ack."""
    from unittest.mock import MagicMock

    from aitune.utils.monitoring.hardware_metrics.parent_receiver import ParentReceiver

    queue = Queue()
    receiver = ParentReceiver(queue)
    mock_ipc = MagicMock()
    mock_ipc.receive.return_value = None
    receiver.ipc = mock_ipc

    path = tmp_path / "snap.csv"
    receiver.snapshot(path, reset_metrics=True)

    mock_ipc.send.assert_called_once_with({"command": "snapshot", "path": str(path), "reset": True})
    mock_ipc.flush.assert_called_once()
    mock_ipc.receive.assert_called_once()


def test_parent_receiver_snapshot_not_started():
    """snapshot() raises RuntimeError when receiver is not started."""
    from aitune.utils.monitoring.hardware_metrics.parent_receiver import ParentReceiver

    queue = Queue()
    receiver = ParentReceiver(queue)

    with pytest.raises(RuntimeError, match="Receiver is not started."):
        receiver.snapshot(Path("/some/path.csv"))


def test_parent_receiver_flow(receiver_and_queue):
    """Test the full flow of ParentReceiver."""
    receiver, queue = receiver_and_queue

    # Test that get_metrics raises error before start
    with pytest.raises(RuntimeError, match="Receiver is not started."):
        receiver.get_metrics()

    receiver.start()
    try:
        # Send some data
        data1 = {"timestamp": 1, "value": 10.0}
        data2 = {"timestamp": 2, "value": 20.0}
        queue.put(data1)
        queue.put(data2)

        # Give some time for the thread to process
        time.sleep(0.5)

        # Get metrics
        df = receiver.get_metrics()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert df.iloc[0]["value"] == 10.0
        assert df.iloc[1]["value"] == 20.0

        # Send more data
        data3 = {"timestamp": 3, "value": 30.0}
        queue.put(data3)
        time.sleep(0.5)

        df_final = receiver.get_metrics()
        assert len(df_final) == 3

    finally:
        if receiver.ipc is not None:
            receiver.stop()
