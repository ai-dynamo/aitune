# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise decorated-forward checkpoint loading with and without a fresh interpreter."""

import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest
import torch

import aitune.torch as ait
from tests.functional_ci.backends import BACKENDS, BackendCase, with_backends
from tests.functional_ci.lifecycle import pinned_strategy, selected_backend
from tests.toy_models.decorated_forward_models import ToyDecoratedSpeechModel


@pytest.mark.matrix("aot_tuning.models.decorated-forward", "modes.aot-save-load")
@with_backends("base")
def test_decorated_forward_checkpoint(case: BackendCase, device: torch.device, tmp_path):
    """Inspect, tune, and save; clear Dynamo caches and load into an uninspected model."""
    # Exercise the same model on every backend, with no fallback or expected failures,
    # so each result shows whether that backend supports the complete lifecycle.
    checkpoint = tmp_path / "decorated.ait"
    backend_type, samples, expected = _save_decorated_checkpoint(case, device, checkpoint)
    torch._dynamo.reset()

    # Load without inspection; different weights also verify checkpoint state restoration.
    fresh_model = ToyDecoratedSpeechModel(seed=1).to(device).eval()
    restored = ait.load(fresh_model, checkpoint)
    assert isinstance(restored, ait.Module)
    selected_backend(restored, backend_type)
    with torch.no_grad():
        for sample, reference in zip(samples, expected, strict=True):
            case.tolerance.assert_close(restored(sample).cpu(), reference)


@pytest.mark.matrix("aot_tuning.models.decorated-forward/{backend_id}", "modes.aot-save-load/{backend_id}")
@pytest.mark.parametrize("backend_id", ["inductor-jit", "torch-tensorrt-jit"])
def test_decorated_forward_checkpoint_fresh_process(backend_id: str, device: torch.device, tmp_path):
    checkpoint = tmp_path / "decorated.ait"
    _, samples, expected = _save_decorated_checkpoint(BACKENDS[backend_id], device, checkpoint)
    evidence = tmp_path / "inference.pt"
    torch.save({"inputs": [sample.cpu() for sample in samples], "expected": expected}, evidence)
    script = dedent("""\
        import sys
        import torch
        import aitune.torch as ait
        from tests.functional_ci.backends import BACKENDS
        from tests.functional_ci.lifecycle import selected_backend
        from tests.toy_models.decorated_forward_models import ToyDecoratedSpeechModel

        checkpoint, evidence, backend_id, device = sys.argv[1:]
        case = BACKENDS[backend_id]
        recorded = torch.load(evidence, map_location="cpu", weights_only=True)
        model = ToyDecoratedSpeechModel(seed=1).to(device).eval()
        restored = ait.load(model, checkpoint)
        assert isinstance(restored, ait.Module)
        selected_backend(restored, type(case.factory()))
        with torch.no_grad():
            for sample, expected in zip(recorded["inputs"], recorded["expected"], strict=True):
                case.tolerance.assert_close(restored(sample.to(device)).cpu(), expected)
    """)
    completed = subprocess.run(
        [sys.executable, "-c", script, str(checkpoint), str(evidence), backend_id, str(device)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, (
        f"Fresh-process checkpoint load or inference failed ({completed.returncode})\n"
        f"{completed.stdout}\n{completed.stderr}"
    )


def _save_decorated_checkpoint(case: BackendCase, device: torch.device, checkpoint: Path):
    backend = case.factory()
    model = ToyDecoratedSpeechModel().to(device).eval()
    value = torch.arange(16, dtype=torch.float32, device=device).reshape(2, 8) / 16
    validation_values = (value, value + 0.5)
    with torch.no_grad():
        expected = [model(sample).cpu() for sample in validation_values]

    inspected = ait.inspect(model, [value], number_of_iterations=1, warmup_iterations=1)
    tuned = ait.wrap(model, inspected.get_modules(), strategy=pinned_strategy(backend))
    assert isinstance(tuned, ait.Module)
    ait.tune(
        tuned,
        list(value.unbind(0)),
        batch_sizes=[value.shape[0]],
        device=device,
        ignore_failing_modules=False,
    )
    selected_backend(tuned, type(backend))
    with torch.no_grad():
        for sample, reference in zip(validation_values, expected, strict=True):
            case.tolerance.assert_close(tuned(sample).cpu(), reference)

    ait.save(tuned, checkpoint)
    tuned.deactivate()
    return type(backend), validation_values, expected
