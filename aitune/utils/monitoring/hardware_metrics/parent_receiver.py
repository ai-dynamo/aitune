# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parent receiver for hardware metrics."""

import multiprocessing.queues
import threading
from pathlib import Path
from queue import Empty, Queue

import pandas as pd

from aitune.utils.monitoring.hardware_metrics.ipc import ParentIpc


class ParentReceiver:
    """Receiver which gets metrics from queue and sends them to child process.

    Communication uses pickle to serialize and deserialize data and sends/receives data via a stdin/stdout pipe.

    If the main process dies, the child process will dump the collected metrics to a CSV file.
    If the main process is still alive and requests the metrics back, the child process will send them as pandas DataFrame so
    that it takes less memory to send them back.

    Args:
        queue: The queue to send the metrics to.
    """

    def __init__(self, queue: Queue | multiprocessing.queues.Queue):
        """Initializes the parent receiver."""
        self.queue = queue
        self.keep_running = threading.Event()
        self.ipc = None
        self.lock = threading.Lock()

    def start(self):
        """Starts the receiver."""
        self.keep_running.set()
        self.ipc = ParentIpc(child_script="child_receiver.py")
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        """Runs the receiver.

        This method is run in a separate thread.
        """
        while self.keep_running.is_set():
            try:
                data = self.queue.get(timeout=1.0)
                with self.lock:
                    self.ipc.send(data)
            except Empty:
                continue

    def stop(self):
        """Stops the receiver."""
        if self.ipc is None:
            raise RuntimeError("Receiver is not started.")

        self.keep_running.clear()
        self.thread.join()
        with self.lock:
            self.ipc.send("quit")
            self.ipc.flush()
        self.ipc = None

    def snapshot(self, path: Path, reset_metrics: bool = True) -> None:
        """Dumps currently collected metrics to a file.

        Args:
            path: Path to write the CSV file.
            reset_metrics: If True, clears the accumulated metrics after writing.

        Raises:
            RuntimeError: If the receiver is not started.
        """
        with self.lock:
            if self.ipc is None:
                raise RuntimeError("Receiver is not started.")
            self.ipc.send({"command": "snapshot", "path": str(path), "reset": reset_metrics})
            self.ipc.flush()
            self.ipc.receive()  # wait for acknowledgment

    def get_metrics(self) -> pd.DataFrame:
        """Gets the metrics.

        Returns:
            The metrics as a pandas DataFrame.

        Raises:
            RuntimeError: If the receiver is not started.

        """
        if self.ipc is None:
            raise RuntimeError("Receiver is not started.")
        with self.lock:
            self.ipc.send("send_results_back")
            self.ipc.flush()
            return self.ipc.receive()
