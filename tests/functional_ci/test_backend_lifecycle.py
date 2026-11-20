# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend lifecycle and exported-artifact cases on a toy MLP."""

from functools import partial
from typing import cast

import pytest
import torch

import aitune.torch as ait
from aitune.records import DType
from aitune.torch.backend.tensorrt.tensorrt_backend import TensorRTProfileMode
from aitune.torch.config import config
from aitune.torch.dataloader import DynamicShapeDataset
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.config import config as jit_config
from aitune.torch.jit.patched_module import ModuleState, PatchedModule
from aitune.torch.jit.patcher import prepare_for_jit_tuning
from aitune.torch.module.tuned_module import TunedModule
from aitune.torch.utils.tensor import format_tensor_name
from tests.functional_ci.artifacts import run_exported_artifact
from tests.functional_ci.backends import BackendCase, tensorrt_backend, with_backends
from tests.functional_ci.lifecycle import TENSORRT_FP32, assert_matches_eager, pinned_strategy, run_lifecycle
from tests.functional_ci.models import IN_FEATURES, OUT_FEATURES, make_image_model, make_input, make_mlp, seeded
from tests.functional_ci.test_jit import JitMLP


@pytest.mark.matrix(
    "aot_tuning.models.tensor-inputs",
    "aot_tuning.models.static-shapes",
    "aot_tuning.models.single-graph",
    "other.input-float32",
    "modes.aot-tune",
    "modes.aot-save-load",
    "topology.{backend}/single-gpu",
    "topology.inspect/single-gpu",
    "strategies.one-backend/aot-tune",
)
@with_backends("base")
def test_backend_lifecycle(case: BackendCase, device: torch.device, tmp_path):
    """Tune, infer, save, and reload the toy MLP."""
    run_lifecycle(case.factory(), partial(make_mlp, device), make_input(device), tmp_path / "model.ait", case.tolerance)


@pytest.mark.matrix("jit_tuning.models.static-shapes")
@with_backends("base")
def test_backend_lifecycle_jit(case: BackendCase, device: torch.device):
    """Tune from a single batch size and serve the same static shape."""
    backend = case.factory()
    jit_config.mode = JITMode.TUNE_EAGER
    jit_config.device = device
    jit_config.min_samples = 1
    jit_config.batch_axis_required = False
    jit_config.strategy = pinned_strategy(backend)
    reference = make_mlp(device)
    with prepare_for_jit_tuning():
        model = JitMLP(make_mlp(device))

    with torch.no_grad():
        model(make_input(device, batch_size=2))
    (head,) = PatchedModule.heads
    assert head._state == ModuleState.TUNED
    (selected,) = cast(TunedModule, head._wrapper).backends.values()
    assert type(selected) is type(backend)
    assert_matches_eager(model, reference, device, case.tolerance, batch_sizes=(2,))


@pytest.mark.matrix("modes.aot-export")
@with_backends("export")
def test_exported_artifact(case: BackendCase, device: torch.device, tmp_path):
    """The tuned module exports a deployable artifact that runs natively, without AITune, and matches eager."""
    tuned = ait.Module(make_mlp(device), "mlp", strategy=pinned_strategy(case.factory()))
    ait.tune(tuned, list(make_input(device).unbind(0)), batch_sizes=[1, 2], device=device)
    artifact = tuned.artifact()
    assert artifact.model.format == case.artifact_format
    assert len(artifact.inputs) == len(artifact.outputs) == 1
    for spec, features in zip(artifact.inputs + artifact.outputs, (IN_FEATURES, OUT_FEATURES), strict=True):
        assert spec.dtype is DType.FLOAT32
        assert spec.min_shape == (1, features)
        assert spec.max_shape == (2, features)
    assert artifact.max_batch_size == 2

    def run_native(value: torch.Tensor) -> torch.Tensor:
        (output,) = run_exported_artifact(artifact, tmp_path / "exported", [value])
        return output

    assert_matches_eager(run_native, make_mlp(device), device, case.tolerance, batch_sizes=(1, 2))

    checkpoint = tmp_path / "model.ait"
    ait.save(tuned, checkpoint)
    tuned.deactivate()
    restored = ait.load(make_mlp(device, seed=1), checkpoint)
    restored_artifact = restored.artifact()
    assert restored_artifact.inputs == artifact.inputs
    assert restored_artifact.outputs == artifact.outputs
    assert restored_artifact.runtime == artifact.runtime
    assert restored_artifact.model.format == artifact.model.format
    assert restored_artifact.model.metadata == artifact.model.metadata
    assert restored_artifact.model.additional_files == artifact.model.additional_files
    assert restored_artifact.model.path != artifact.model.path

    def run_restored_native(value: torch.Tensor) -> torch.Tensor:
        (output,) = run_exported_artifact(restored_artifact, tmp_path / "restored-export", [value])
        return output

    assert_matches_eager(run_restored_native, make_mlp(device), device, case.tolerance, batch_sizes=(1, 2))


@pytest.mark.matrix("other.tensorrt-multi-profile")
def test_tensorrt_multi_profile_artifact(device: torch.device, tmp_path, monkeypatch):
    """Sampled and fallback profiles survive out-of-profile errors, native export, and checkpoint reload."""
    monkeypatch.setattr(config, "max_num_samples_stored", 4)
    backend = tensorrt_backend(profiles=TensorRTProfileMode.SAMPLES_USED)
    tuned = ait.Module(make_image_model(device), "images", strategy=pinned_strategy(backend))
    reference = make_image_model(device)
    with seeded(device, 1):
        samples = [torch.randn(3, size, size, device=device) for size in (8, 16)]
    ait.tune(tuned, DynamicShapeDataset(samples), batch_sizes=[2, 4], device=device)

    with torch.no_grad():
        for sample in samples:
            for batch_size in (4, 3):
                value = sample.repeat(batch_size, 1, 1, 1)
                TENSORRT_FP32.assert_close(tuned(value), reference(value))
        with pytest.raises(RuntimeError):
            tuned(samples[0].repeat(5, 1, 1, 1))
        value = samples[1].repeat(2, 1, 1, 1)
        TENSORRT_FP32.assert_close(tuned(value), reference(value))

    artifact = tuned.artifact()
    assert artifact.model.format == "tensorrt_plan"
    input_name = format_tensor_name("input", "input")
    assert artifact.input_names == (input_name,)
    assert artifact.output_names == (format_tensor_name((), "output"),)
    (input_spec,) = artifact.inputs
    (output_spec,) = artifact.outputs
    assert input_spec.dtype is output_spec.dtype is DType.FLOAT32
    assert input_spec.min_shape == (2, 3, 8, 8)
    assert input_spec.max_shape == (4, 3, 16, 16)
    assert output_spec.min_shape == (2, 4)
    assert output_spec.max_shape == (4, 4)
    assert input_spec.batch_axis == output_spec.batch_axis == 0
    assert artifact.max_batch_size is None
    profiles = artifact.model.metadata["optimization_profiles"]
    assert artifact.model.metadata["optimization_profile_count"] == len(profiles) == 5
    assert {profile[input_name]["min_shape"] for profile in profiles[:-1]} == {
        (batch_size, 3, size, size) for batch_size in (2, 4) for size in (8, 16)
    }
    for profile in profiles[:-1]:
        assert tuple(profile) == artifact.input_names
        bounds = profile[input_name]
        assert bounds["min_shape"] == bounds["opt_shape"] == bounds["max_shape"]
    assert profiles[-1] == {
        input_name: {"min_shape": (2, 3, 8, 8), "opt_shape": (4, 3, 16, 16), "max_shape": (4, 3, 16, 16)}
    }

    import tensorrt as trt

    plan = artifact.model.export_files(tmp_path / "export" / "model.plan")
    # TensorRT needs the logger and the runtime to outlive the engine.
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan.read_bytes())
    assert engine is not None
    names = tuple(engine.get_tensor_name(index) for index in range(engine.num_io_tensors))
    assert names == artifact.input_names + artifact.output_names
    assert engine.num_optimization_profiles == len(profiles)
    for index, profile in enumerate(profiles):
        bounds = profile[input_name]
        expected = (bounds["min_shape"], bounds["opt_shape"], bounds["max_shape"])
        assert tuple(tuple(shape) for shape in engine.get_tensor_profile_shape(input_name, index)) == expected

    checkpoint = tmp_path / "images.ait"
    ait.save(tuned, checkpoint)
    tuned.deactivate()
    restored = ait.load(make_image_model(device, seed=1), checkpoint)
    restored_artifact = restored.artifact()
    assert restored_artifact.inputs == artifact.inputs
    assert restored_artifact.outputs == artifact.outputs
    assert restored_artifact.runtime == artifact.runtime
    assert restored_artifact.model.format == artifact.model.format
    assert restored_artifact.model.metadata == artifact.model.metadata
    with torch.no_grad():
        for sample in samples:
            for batch_size in (2, 3, 4):
                value = sample.repeat(batch_size, 1, 1, 1)
                TENSORRT_FP32.assert_close(restored(value), reference(value))
