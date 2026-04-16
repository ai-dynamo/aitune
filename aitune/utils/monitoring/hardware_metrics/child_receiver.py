# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entry point for the child subprocess that accumulates hardware metrics forwarded by the parent process.

On a normal shutdown the child sends the accumulated data back as a DataFrame. If the parent process
terminates unexpectedly (detected as an IPC error), the child dumps the metrics to a CSV file instead.
"""

import logging
from logging import basicConfig

import pandas as pd

from aitune.utils.monitoring.hardware_metrics.ipc import ChildIpc, IpcError
from aitune.utils.monitoring.setup_hardware_metrics import dump_metrics


def run_child_receiver():
    """Run the child receiver.

    In case parent process dies (which is detected by checking the pipe for EOF), it dumps the collected metrics
    to a CSV file. If parent process is still alive and requests the metrics back, sends them as pandas DataFrame so
    that it takes less memory to send them back.

    Communication uses pickle to serialize and deserialize data and sends/receives data via a stdin/stdout pipe.
    """
    ipc = ChildIpc()
    collected_metrics = []
    reason = None

    while True:
        try:
            data = ipc.receive()
            if data == "send_results_back":
                df = pd.DataFrame(collected_metrics)  # convert to DataFrame to save memory
                ipc.send(df)
            elif isinstance(data, dict) and data.get("command") == "snapshot":
                df = pd.DataFrame(collected_metrics)
                df.to_csv(data["path"], index=False)
                if data.get("reset", True):
                    collected_metrics.clear()
                ipc.send(None)  # acknowledgment
            elif data == "quit":
                break
            else:
                collected_metrics.append(data)
        except IpcError:
            reason = "Parent process terminated."
            break
        except Exception as e:
            reason = f"Unexpected exception: {e}"
            break

    if reason:
        basicConfig(level=logging.INFO, force=True)  # initialize logging
        logging.error("There was an error while tuning: %s", reason)
        df = pd.DataFrame(collected_metrics)  # convert to DataFrame to save memory
        dump_metrics(df)


if __name__ == "__main__":
    run_child_receiver()
