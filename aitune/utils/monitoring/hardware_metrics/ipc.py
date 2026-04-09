# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inter-process communication utilities."""

import pickle
import struct
import subprocess
import sys
from pathlib import Path


class IpcError(Exception):
    """Exception for IPC errors."""

    pass


def _read_exactly(stream, n: int) -> bytes:
    """Read exactly n bytes from stream, looping until all bytes are received."""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return bytes(buf)
        buf += chunk
    return bytes(buf)


class ParentIpc:
    """Simple inter-process communication class to send and receive data between parent and child processes.

    This is implementation for parent process. It uses stdin/stdout pipes to send and receive data which is
    packed using pickle.

    Note: there is assumption that parent will sent lots of small data, that is why flush_interval is used so that
    underlying operating system can buffer data and send it in larger chunks which improves throughput.
    """

    def __init__(self, child_script, flush_interval=10):
        """Initializes the parent IPC.

        Args:
            child_script: The script to run in the child process.
            flush_interval: The interval to flush the data to the child process in number of messages.

        Note: the child has clean env variables to avoid conflicts with the parent process.
        """
        here = Path(__file__).parent

        self.child = subprocess.Popen(
            [sys.executable, here / child_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            start_new_session=True,
            env={},  # reset child environment variables
        )
        if self.child.poll() is not None:
            raise IpcError(f"Child process failed to start. Return code: {self.child.returncode}")

        self.counter = 0
        self.flush_interval = flush_interval

    def send(self, data):
        """Send data to the child process.

        Args:
            data: The data to send to the child process.

        Returns:
            The number of bytes sent.
        """
        serialized = pickle.dumps(data)
        header = struct.pack("!I", len(serialized))
        payload = header + serialized
        self.child.stdin.write(payload)
        self.counter += 1
        if self.counter % self.flush_interval == 0:
            self.child.stdin.flush()
        return len(payload)

    def flush(self):
        """Flush the data to the child process."""
        try:
            self.child.stdin.flush()
        except BrokenPipeError:
            ...

    def receive(self):
        """Receive data from the child process."""
        header = _read_exactly(self.child.stdout, 4)
        if header:
            size = struct.unpack("!I", header)[0]
            return pickle.loads(_read_exactly(self.child.stdout, size))
        return None


class ChildIpc:
    """Simple inter-process communication class to send and receive data between parent and child processes.

    This is implementation for child process. It uses stdin/stdout pipes to send and receive data which is
    packed using pickle.
    """

    def __init__(self):
        """Initializes the child IPC."""
        self.stdin_raw = sys.stdin.buffer
        self.stdout_raw = sys.stdout.buffer

    def receive(self):
        """Receive data from the parent process.

        Returns:
            The data received from the parent process.

        Raises:
            ValueError: If the parent process closed the pipe.
        """
        # Read the 4-byte integer header
        header = _read_exactly(self.stdin_raw, 4)
        if not header:
            raise IpcError("Parent process closed the pipe.")

        size = struct.unpack("!I", header)[0]
        data = _read_exactly(self.stdin_raw, size)
        return pickle.loads(data)

    def send(self, data):
        """Send data to the parent process.

        Args:
            data: The data to send to the parent process.

        Returns:
            The number of bytes sent.
        """
        serialized = pickle.dumps(data)
        header = struct.pack("!I", len(serialized))
        payload = header + serialized
        self.stdout_raw.write(payload)
        self.stdout_raw.flush()
        return len(payload)
