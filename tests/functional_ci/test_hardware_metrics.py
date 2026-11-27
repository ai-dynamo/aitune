# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardware metrics from tuned inference and snapshot boundaries."""

from csv import DictReader
from pathlib import Path
from time import monotonic, sleep

import pytest
import torch

import aitune.torch as ait
from aitune.utils.monitoring import annotate, disable_hardware_metrics, enable_hardware_metrics, snapshot
from aitune.utils.monitoring import setup_hardware_metrics as metrics_setup
from tests.functional_ci.backends import BACKENDS
from tests.functional_ci.lifecycle import assert_matches_eager, pinned_strategy
from tests.functional_ci.models import make_input, make_mlp


def _infer_timestamps(path: Path, module_name: str, backend_name: str) -> set[str]:
    with path.open(newline="") as file:
        return {
            row["timestamp"]
            for row in DictReader(file)
            if (row["scope"], row["module_name"], row["backend"]) == ("infer", module_name, backend_name)
        }


def _wait_for_inference_samples(path: Path) -> None:
    # The receiver forwards samples asynchronously; this scope marks the end of the inference samples.
    with annotate(name="inference-complete"):
        pass
    deadline = monotonic() + 10
    while monotonic() < deadline:
        snapshot(path, reset_metrics=False)
        with path.open(newline="") as file:
            if any(row["scope"] == "inference-complete" for row in DictReader(file)):
                return
        sleep(0.01)
    pytest.fail("Hardware metrics receiver did not write the inference completion sample")


@pytest.mark.matrix("other.hardware-metrics")
def test_hardware_metrics(device: torch.device, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    eager = BACKENDS["torch-eager"]
    backend = eager.factory()
    module_name = "metrics-mlp"
    reference = make_mlp(device)
    tuned = ait.Module(make_mlp(device), module_name, strategy=pinned_strategy(backend))
    value = make_input(device)
    ait.tune(tuned, list(value.unbind(0)), batch_sizes=[value.shape[0]], device=device)

    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    dump_path = tmp_path / "hardware_metrics.csv"
    # AITUNE_HARDWARE_METRICS_PATH is read at import time.
    monkeypatch.setattr(metrics_setup, "HARDWARE_METRICS_PATH", str(dump_path))
    enable_hardware_metrics()
    try:
        assert_matches_eager(tuned, reference, device, eager.tolerance)
        _wait_for_inference_samples(first_path)
        snapshot(first_path)
        first = _infer_timestamps(first_path, module_name, backend.describe())
        assert first

        assert_matches_eager(tuned, reference, device, eager.tolerance)
        _wait_for_inference_samples(second_path)
        second = _infer_timestamps(second_path, module_name, backend.describe())
        assert second
        assert first.isdisjoint(second)
    finally:
        disable_hardware_metrics()

    assert _infer_timestamps(dump_path, module_name, backend.describe()) == second
