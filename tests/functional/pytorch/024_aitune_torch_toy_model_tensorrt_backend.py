# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from logging import DEBUG, basicConfig, getLogger
from pathlib import Path
from tempfile import TemporaryDirectory

import tensorrt as trt
import torch
from polygraphy.backend.trt import TrtRunner

from aitune.records import DeploymentArtifact, DType
from aitune.torch.backend.tensorrt.tensorrt_backend import (
    ProfileMode,
    TensorRTBackend,
    TensorRTBackendConfig,
)
from aitune.torch.config import config as global_config
from aitune.torch.dataloader import DynamicShapeDataset
from aitune.torch.module.wrapper_module import Module
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy
from aitune.torch.tuning import load, save, tune
from aitune.torch.utils.tensor import format_tensor_name

logger = getLogger(Path(__file__).stem)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.1


def _check_exported_artifact(artifact, destination):
    assert artifact.model.export_files(destination) == destination
    assert destination.read_bytes() == artifact.model.path.read_bytes()
    assert tuple(destination.parent.iterdir()) == (destination,)

    # Load the exported plan independently of AITune's runtime and checkpoint sidecars.
    trt_logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(destination.read_bytes())
    assert engine is not None
    names = tuple(engine.get_tensor_name(index) for index in range(engine.num_io_tensors))
    assert (
        tuple(name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT) == artifact.input_names
    )
    assert (
        tuple(name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT)
        == artifact.output_names
    )
    assert all(engine.get_tensor_dtype(name) == trt.float32 for name in names)
    profiles = artifact.model.metadata["optimization_profiles"]
    assert engine.num_optimization_profiles == len(profiles)

    with TrtRunner(engine) as runner:
        for index, profile in enumerate(profiles):
            runner.set_profile(index)
            assert tuple(profile) == artifact.input_names
            bounds = profile[artifact.input_names[0]]
            assert tuple(tuple(shape) for shape in engine.get_tensor_profile_shape(artifact.input_names[0], index)) == (
                bounds["min_shape"],
                bounds["opt_shape"],
                bounds["max_shape"],
            )
            value = torch.randn(bounds["opt_shape"])
            expected = ToyModel()(value)
            outputs = runner.infer({artifact.input_names[0]: value.numpy()})
            assert tuple(outputs) == artifact.output_names
            torch.testing.assert_close(
                torch.from_numpy(outputs[artifact.output_names[0]]), expected, rtol=1e-5, atol=1e-6
            )


def testing_multi_profile_with_samples():
    device = "cuda"
    dtype = torch.float32

    # create a model that process images of size 224x224 and 448x448
    model = ToyModel()
    model.to(device, dtype=dtype)
    model.eval()

    # create two samples images of size 224x224 and 448x448
    data1 = torch.randn((3, 224, 224), device=device).to(dtype)
    data2 = torch.randn((3, 448, 448), device=device).to(dtype)

    # set max_num_samples_stored to 4 to generate all profiles 2 samples x 2 batch sizes
    global_config.max_num_samples_stored = 4

    # configure the backend to use multi-profile mode, and auto generate profiles from samples used for tuning
    backend = TensorRTBackend(TensorRTBackendConfig(profiles=ProfileMode.SAMPLES_USED))
    strategy = OneBackendStrategy(backend)
    strategy.enable_performance_validation(False)
    strategy.enable_find_max_batch_size(False)
    module = Module(
        model,
        "toy-model",
        strategy=strategy,
    )

    # tune the model with the two samples, need to use DynamicShapeDataset as samples are different shapes
    tune(module, DynamicShapeDataset([data1, data2]), batch_sizes=[2, 8], device=device)

    # testing model with different samples and batch sizes
    module(data1.repeat(8, 1, 1, 1))
    module(data2.repeat(8, 1, 1, 1))

    try:
        module(data1.repeat(4, 1, 1, 1))
    except RuntimeError:
        pass  # expected exception when passing sample with shape that is not in any profile
    else:
        raise AssertionError("Expected exception when passing sample with shape that is not in any profile")

    # still runs after error
    module(data2.repeat(2, 1, 1, 1))

    # Check the public artifact against the four profiles selected during tuning.
    artifact = module.artifact()
    assert isinstance(artifact, DeploymentArtifact)
    assert artifact.model.format == "tensorrt_plan"
    assert artifact.model.path.is_file()
    assert artifact.model.additional_files == ()
    assert artifact.model.files == (artifact.model.path,)
    assert artifact.runtime.name == "tensorrt"
    assert artifact.runtime.options == {
        "use_cuda_graphs": True,
        "max_cuda_graphs": 8,
        "cuda_graph_cache_policy": "lfu",
    }
    input_name = format_tensor_name("x", "input")
    assert artifact.input_names == (input_name,)
    assert len(artifact.outputs) == 1
    for spec in artifact.inputs + artifact.outputs:
        assert spec.dtype is DType.FLOAT32
        assert spec.min_shape == (2, 3, 224, 224)
        assert spec.max_shape == (8, 3, 448, 448)
        assert spec.batch_axis == 0
    assert artifact.max_batch_size is None  # These profiles do not support batch size one.
    profiles = artifact.model.metadata["optimization_profiles"]
    assert artifact.model.metadata["optimization_profile_count"] == len(profiles) == 4
    assert {profile[input_name]["min_shape"] for profile in profiles} == {
        (8, 3, 448, 448),
        (2, 3, 448, 448),
        (8, 3, 224, 224),
        (2, 3, 224, 224),
    }
    for profile in profiles:
        bounds = profile[input_name]
        assert bounds["min_shape"] == bounds["opt_shape"] == bounds["max_shape"]

    with TemporaryDirectory(prefix="aitune-tensorrt-artifact-") as directory:
        root = Path(directory)
        _check_exported_artifact(artifact, root / "export" / "renamed.plan")
        checkpoint = root / "model.ait"
        save(module, checkpoint)
        module.deactivate()

        restored = load(ToyModel().eval(), checkpoint)
        restored_artifact = restored.artifact()
        assert restored_artifact.model.path != artifact.model.path
        assert restored_artifact.model.format == artifact.model.format
        assert restored_artifact.model.additional_files == artifact.model.additional_files
        assert restored_artifact.model.metadata == artifact.model.metadata
        assert restored_artifact.inputs == artifact.inputs
        assert restored_artifact.outputs == artifact.outputs
        assert restored_artifact.runtime == artifact.runtime
        _check_exported_artifact(restored_artifact, root / "restored-export" / "model.plan")
        value = data2.repeat(2, 1, 1, 1)
        torch.testing.assert_close(restored(value), value * 0.1, rtol=1e-5, atol=1e-6)

    MODULE_REGISTRY.clear()


if __name__ == "__main__":
    basicConfig(level=DEBUG, force=True)
    testing_multi_profile_with_samples()
