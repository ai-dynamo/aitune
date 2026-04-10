# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# scope = "always"
#
# [environment]
# AITUNE_HARDWARE_METRICS = "1"
# AITUNE_HARDWARE_METRICS_PATH = "hardware_metrics.csv"
# ///

import os
from logging import INFO, basicConfig
from pathlib import Path
from time import sleep

import pandas as pd

from aitune.global_context import BACKEND_CONTEXT_KEY, MODULE_CONTEXT_KEY, global_context
from aitune.utils.monitoring import annotate
from aitune.utils.monitoring.setup_hardware_metrics import dump_metrics, get_default_collector, get_hardware_metrics


def test_hardware_metrics_path():
    assert os.environ.get("AITUNE_HARDWARE_METRICS") == "1"
    assert os.environ.get("AITUNE_HARDWARE_METRICS_PATH", "hardware_metrics.csv") == "hardware_metrics.csv"
    get_default_collector()

    @annotate(name="custom_foo")
    def foo():
        with global_context:
            global_context.set(MODULE_CONTEXT_KEY, "foo_module")
            global_context.set(BACKEND_CONTEXT_KEY, "foo_backend")
            sleep(0.2)

    @annotate()
    def bar():
        with global_context:
            global_context.set(MODULE_CONTEXT_KEY, "bar_module")
            global_context.set(BACKEND_CONTEXT_KEY, "bar_backend")
            sleep(0.2)

    # Call the functions to collect metrics
    foo()
    bar()

    metrics = get_hardware_metrics()
    assert metrics is not None

    dump_metrics(metrics)
    assert Path("hardware_metrics.csv").exists()

    df = pd.read_csv("hardware_metrics.csv")
    df = df.dropna(subset=["module_name", "backend"])
    assert set(df["scope"].unique()) == {"custom_foo", "bar"}
    assert set(df["module_name"].unique()) == {"foo_module", "bar_module"}
    assert set(df["backend"].unique()) == {"foo_backend", "bar_backend"}


if __name__ == "__main__":
    basicConfig(level=INFO, force=True)
    test_hardware_metrics_path()
