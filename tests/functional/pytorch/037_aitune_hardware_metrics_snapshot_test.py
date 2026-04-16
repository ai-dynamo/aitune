# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# scope = "always"
#
# [environment]
# AITUNE_HARDWARE_METRICS = "1"
# ///

import tempfile
from logging import INFO, basicConfig
from pathlib import Path
from time import sleep

import pandas as pd

from aitune.global_context import BACKEND_CONTEXT_KEY, MODULE_CONTEXT_KEY, global_context
from aitune.utils.monitoring import annotate, snapshot
from aitune.utils.monitoring.setup_hardware_metrics import get_default_collector


def test_hardware_metrics_snapshot():
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

    @annotate(name="custom_baz")
    def baz():
        with global_context:
            global_context.set(MODULE_CONTEXT_KEY, "baz_module")
            global_context.set(BACKEND_CONTEXT_KEY, "baz_backend")
            sleep(0.2)

    with tempfile.TemporaryDirectory() as tmp:
        snap1 = Path(tmp) / "snap1.csv"
        snap2 = Path(tmp) / "snap2.csv"
        snap3 = Path(tmp) / "snap3.csv"

        # Phase 1: snapshot with reset (default) — only foo data captured
        foo()
        snapshot(snap1)

        assert snap1.exists()
        df1 = pd.read_csv(snap1).dropna(subset=["module_name", "backend"])
        assert set(df1["scope"].unique()) == {"custom_foo"}
        assert set(df1["module_name"].unique()) == {"foo_module"}
        assert set(df1["backend"].unique()) == {"foo_backend"}

        # Phase 2: snapshot without reset — bar data preserved for next snapshot
        bar()
        snapshot(snap2, reset_metrics=False)

        assert snap2.exists()
        df2 = pd.read_csv(snap2).dropna(subset=["module_name", "backend"])
        assert set(df2["scope"].unique()) == {"bar"}
        assert set(df2["module_name"].unique()) == {"bar_module"}
        assert set(df2["backend"].unique()) == {"bar_backend"}

        # Phase 3: bar metrics still accumulated; baz added; snapshot resets again
        baz()
        snapshot(snap3)

        assert snap3.exists()
        df3 = pd.read_csv(snap3).dropna(subset=["module_name", "backend"])
        assert set(df3["scope"].unique()) == {"bar", "custom_baz"}
        assert set(df3["module_name"].unique()) == {"bar_module", "baz_module"}
        assert set(df3["backend"].unique()) == {"bar_backend", "baz_backend"}


if __name__ == "__main__":
    basicConfig(level=INFO, force=True)
    test_hardware_metrics_snapshot()
