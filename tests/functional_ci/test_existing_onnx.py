# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tune an existing ONNX file, with inline or external weights, through `OnnxModule`."""

from functools import partial
from pathlib import Path

import onnx
import pytest
import torch

import aitune.torch as ait
from aitune.torch.module.onnx_module import OnnxModule
from tests.functional_ci.artifacts import run_exported_artifact
from tests.functional_ci.backends import BackendCase, with_backends
from tests.functional_ci.lifecycle import assert_matches_eager, pinned_strategy, selected_backend
from tests.functional_ci.models import make_input, make_mlp


def export_mlp(device: torch.device, directory: Path, external_data: bool) -> Path:
    """Export the toy MLP with input `x`, output `y`, and a dynamic batch axis."""
    path = directory / "mlp.onnx"
    torch.onnx.export(
        make_mlp(device),
        (make_input(device),),
        str(path),
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
        dynamo=False,
    )
    if not external_data:
        return path
    external = directory / "external" / "mlp.onnx"
    external.parent.mkdir()
    onnx.save_model(
        onnx.load(str(path)), str(external), save_as_external_data=True, location="weights.bin", size_threshold=0
    )
    assert (external.parent / "weights.bin").is_file()
    return external


def run_native_onnx_artifact(artifact, export_dir, value):
    (output,) = run_exported_artifact(artifact, export_dir, [value])
    return {"y": output}


@pytest.mark.matrix("modes.aot-existing-onnx", "modes.aot-save-load", "modes.aot-export")
@pytest.mark.parametrize("external_data", [False, True], ids=["inline", "external-data"])
@with_backends("onnx_input")
def test_existing_onnx(case: BackendCase, external_data: bool, device: torch.device, tmp_path):
    """Named ONNX I/O and external weights survive tuning, checkpoint reload, and native artifact execution."""
    backend = case.factory()
    source = export_mlp(device, tmp_path, external_data)
    tuned = ait.Module(OnnxModule(source), "onnx-mlp", strategy=pinned_strategy(backend))
    samples = [{"x": row} for row in make_input(device, batch_size=4).unbind(0)]
    ait.tune(tuned, samples, batch_sizes=[1, 4], device=device)

    selected_backend(tuned, type(backend))
    reference = make_mlp(device)
    # `assert_close` on dicts also checks that `y` is the only output.
    assert_matches_eager(
        lambda x: tuned(x=x), lambda x: {"y": reference(x)}, device, case.tolerance, batch_sizes=(1, 3, 4)
    )

    checkpoint = tmp_path / "existing-onnx.ait"
    ait.save(tuned, checkpoint)
    tuned.deactivate()
    restored = ait.load(OnnxModule(source), checkpoint)
    assert isinstance(restored, ait.Module)
    selected_backend(restored, type(backend))
    assert_matches_eager(
        lambda x: restored(x=x), lambda x: {"y": reference(x)}, device, case.tolerance, batch_sizes=(1, 3, 4)
    )

    artifact = restored.artifact()
    assert artifact.model.format == case.artifact_format

    assert_matches_eager(
        partial(run_native_onnx_artifact, artifact, tmp_path / "packaged"),
        lambda x: {"y": reference(x)},
        device,
        case.tolerance,
        batch_sizes=(1, 3, 4),
    )
