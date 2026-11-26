# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce decorated-forward checkpoint loading in a fresh customer process."""

import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest
import torch

import aitune.torch as ait
from aitune.torch.backend import TorchInductorJitBackend, TorchInductorJitBackendConfig
from tests.functional_ci.lifecycle import Tolerance, pinned_strategy, selected_backend
from tests.toy_models.decorated_forward_models import ToyDecoratedSpeechModel


@pytest.mark.matrix("aot_tuning.models.decorated-forward/inductor-jit", "modes.aot-save-load/inductor-jit")
def test_decorated_forward_checkpoint_fresh_process(device: torch.device, tmp_path):
    """Inspect, tune, and save; load into an uninspected model in a separate interpreter."""
    # This regression targets Dynamo guard generation in Inductor JIT deployment.
    # Pin the backend and reject tuning failures so eager fallback cannot hide it.
    backend = TorchInductorJitBackend(TorchInductorJitBackendConfig(fullgraph=False, dynamic=False, mode="default"))
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
    selected_backend(tuned, TorchInductorJitBackend)
    with torch.no_grad():
        for sample, reference in zip(validation_values, expected, strict=True):
            Tolerance().assert_close(tuned(sample).cpu(), reference)

    checkpoint = tmp_path / "decorated.ait"
    ait.save(tuned, checkpoint)
    tuned.deactivate()
    evidence = tmp_path / "inference.pt"
    torch.save({"inputs": [sample.cpu() for sample in validation_values], "expected": expected}, evidence)

    # A Dynamo reset alone does not reproduce a new interpreter's model state.
    # Do not inspect the restored model: that would hide the descriptor difference.
    script = dedent("""\
        import sys
        import torch
        import aitune.torch as ait
        from aitune.torch.backend import TorchInductorJitBackend
        from tests.functional_ci.lifecycle import Tolerance, selected_backend
        from tests.toy_models.decorated_forward_models import ToyDecoratedSpeechModel

        checkpoint, evidence, device = sys.argv[1:]
        payload = torch.load(evidence, map_location="cpu", weights_only=True)
        # Different weights ensure the check also proves checkpoint state restoration.
        model = ToyDecoratedSpeechModel(seed=1).to(device).eval()
        restored = ait.load(model, checkpoint)
        assert isinstance(restored, ait.Module)
        selected_backend(restored, TorchInductorJitBackend)
        with torch.no_grad():
            for sample, expected in zip(payload["inputs"], payload["expected"], strict=True):
                Tolerance().assert_close(restored(sample.to(device)).cpu(), expected)
    """)
    completed = subprocess.run(
        [sys.executable, "-c", script, str(checkpoint), str(evidence), str(device)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, (
        f"Fresh-process checkpoint loading or inference failed (exit {completed.returncode}).\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
