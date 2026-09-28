# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from pathlib import Path

import pytest
from google.protobuf import text_format
from tritonclient.grpc import model_config_pb2

from aitune import triton as aitriton
from aitune.exceptions import AITunePublicationError
from aitune.records import (
    BoundedTensorSpec,
    DeploymentArtifact,
    DType,
    ModelFiles,
    RuntimeConfig,
)
from aitune.triton import DynamicBatcher, ONNXRuntimeModelConfig, TensorRTModelConfig
from aitune.triton.config.common import tensor_config


def _spec(
    name: str,
    dtype: DType,
    minimum: tuple[int, ...],
    maximum: tuple[int, ...],
) -> BoundedTensorSpec:
    return BoundedTensorSpec(name=name, dtype=dtype, min_shape=minimum, max_shape=maximum)


def _interface() -> tuple[tuple[BoundedTensorSpec, ...], tuple[BoundedTensorSpec, ...]]:
    return (
        (_spec("input_ids", DType.INT64, (1, 8), (8, 512)),),
        (_spec("logits", DType.FLOAT32, (1, 4), (8, 4)),),
    )


def _plan(path: Path, *, profiles: int = 1, use_cuda_graphs: bool = False) -> DeploymentArtifact:
    path.write_bytes(b"TensorRT plan")
    inputs, outputs = _interface()
    return DeploymentArtifact(
        model=ModelFiles(
            format="tensorrt_plan",
            path=path,
            metadata={
                "optimization_profile_count": profiles,
                "optimization_profiles": (
                    {"input_ids": {"min_shape": (1, 8), "opt_shape": (4, 128), "max_shape": (8, 512)}},
                )
                * profiles,
            },
        ),
        inputs=inputs,
        outputs=outputs,
        runtime=RuntimeConfig(name="tensorrt", options={"use_cuda_graphs": use_cuda_graphs}),
    )


def test_publishes_tensorrt_plan_with_bounds_batching_and_profiles(tmp_path):
    artifact = _plan(tmp_path / "source.plan", profiles=2, use_cuda_graphs=True)

    model = aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")

    assert (model / "1" / "model.plan").read_bytes() == b"TensorRT plan"
    config = (model / "config.pbtxt").read_text()
    parsed = text_format.Parse(config, model_config_pb2.ModelConfig())
    assert parsed.name == "encoder"
    assert parsed.max_batch_size == 8
    assert 'platform: "tensorrt_plan"' in config
    assert "max_batch_size: 8" in config
    assert parsed.input[0].name == "input_ids"
    assert parsed.input[0].data_type == model_config_pb2.TYPE_INT64
    assert tuple(parsed.input[0].dims) == (-1,)
    assert tuple(parsed.instance_group[0].profile) == ("0", "1")
    assert parsed.optimization.cuda.graphs
    assert parsed.HasField("dynamic_batching")


def test_artifact_uses_supplied_matching_config(tmp_path):
    artifact = _plan(tmp_path / "source.plan")
    config = TensorRTModelConfig.from_artifact(artifact, name="encoder")
    config.batcher = DynamicBatcher(max_queue_delay_microseconds=100)

    model = aitriton.publish(artifact, path=tmp_path / "repository", config=config)

    parsed = text_format.Parse((model / "config.pbtxt").read_text(), model_config_pb2.ModelConfig())
    assert model.name == "encoder"
    assert parsed.dynamic_batching.max_queue_delay_microseconds == 100
    assert parsed.max_batch_size == 8


def test_artifact_config_can_use_full_tensor_shapes_without_triton_batching(tmp_path):
    artifact = _plan(tmp_path / "source.plan")
    config = TensorRTModelConfig(
        name="encoder",
        max_batch_size=0,
        inputs=tuple(tensor_config(spec, batched=False) for spec in artifact.inputs),
        outputs=tuple(tensor_config(spec, batched=False) for spec in artifact.outputs),
        batcher=None,
        optimization_profile_indices=(0,),
    )

    model = aitriton.publish(artifact, path=tmp_path / "repository", config=config)

    parsed = text_format.Parse((model / "config.pbtxt").read_text(), model_config_pb2.ModelConfig())
    assert parsed.max_batch_size == 0
    assert not parsed.HasField("dynamic_batching")
    assert tuple(parsed.input[0].dims) == (-1, -1)


def test_onnx_artifact_uses_supplied_config_without_triton_batching(tmp_path):
    source = tmp_path / "source.onnx"
    source.write_bytes(b"ONNX graph")
    inputs, outputs = _interface()
    artifact = DeploymentArtifact(
        model=ModelFiles(format="onnx", path=source),
        inputs=inputs,
        outputs=outputs,
        runtime=RuntimeConfig(name="onnxruntime", options={"execution_provider": "cuda"}),
    )
    config = ONNXRuntimeModelConfig(
        name="encoder",
        max_batch_size=0,
        inputs=tuple(tensor_config(spec, batched=False) for spec in inputs),
        outputs=tuple(tensor_config(spec, batched=False) for spec in outputs),
        batcher=None,
        execution_provider="cuda",
    )

    model = aitriton.publish(artifact, path=tmp_path / "repository", config=config)

    parsed = text_format.Parse((model / "config.pbtxt").read_text(), model_config_pb2.ModelConfig())
    assert parsed.max_batch_size == 0
    assert not parsed.HasField("dynamic_batching")
    assert tuple(parsed.input[0].dims) == (-1, -1)


@pytest.mark.parametrize("max_batch_size", [4, 16])
def test_artifact_config_can_override_batch_limit(tmp_path, max_batch_size):
    artifact = _plan(tmp_path / "source.plan")
    config = TensorRTModelConfig.from_artifact(artifact, name="encoder")
    config.max_batch_size = max_batch_size

    model = aitriton.publish(artifact, path=tmp_path / "repository", config=config)

    parsed = text_format.Parse((model / "config.pbtxt").read_text(), model_config_pb2.ModelConfig())
    assert parsed.max_batch_size == max_batch_size


def test_artifact_config_can_omit_scheduler_with_positive_batch_limit(tmp_path):
    artifact = _plan(tmp_path / "source.plan")
    config = TensorRTModelConfig.from_artifact(artifact, name="encoder")
    config.batcher = None

    model = aitriton.publish(artifact, path=tmp_path / "repository", config=config)

    parsed = text_format.Parse((model / "config.pbtxt").read_text(), model_config_pb2.ModelConfig())
    assert parsed.max_batch_size == 8
    assert not parsed.HasField("dynamic_batching")


@pytest.mark.parametrize("mismatch", ["name", "platform"])
def test_artifact_rejects_config_that_does_not_match(tmp_path, mismatch):
    artifact = _plan(tmp_path / "source.plan")
    default = TensorRTModelConfig.from_artifact(artifact, name="encoder")
    if mismatch == "name":
        config = default.model_copy(update={"name": "other"})
        model_name = "encoder"
    elif mismatch == "platform":
        config = ONNXRuntimeModelConfig(
            name="encoder",
            max_batch_size=default.max_batch_size,
            inputs=default.inputs,
            outputs=default.outputs,
            execution_provider="cuda",
        )
        model_name = None
    with pytest.raises(AITunePublicationError, match="config.*artifact|artifact.*config"):
        aitriton.publish(artifact, path=tmp_path / "repository", model_name=model_name, config=config)

    assert not (tmp_path / "repository").exists()


def test_artifact_accepts_supplied_tensor_and_profile_settings(tmp_path):
    artifact = _plan(tmp_path / "source.plan", profiles=2)
    default = TensorRTModelConfig.from_artifact(artifact, name="encoder")
    changed_input = default.inputs[0].model_copy(update={"dims": (16,)})
    config = default.model_copy(update={"inputs": (changed_input,), "optimization_profile_indices": (2,)})

    model = aitriton.publish(artifact, path=tmp_path / "repository", config=config)

    parsed = text_format.Parse((model / "config.pbtxt").read_text(), model_config_pb2.ModelConfig())
    assert tuple(parsed.input[0].dims) == (16,)
    assert tuple(parsed.instance_group[0].profile) == ("2",)


def test_publish_keeps_implicit_batch_axis_for_batch_one_artifact(tmp_path):
    """A batch-one artifact retains Triton's batch axis without a dynamic batcher."""
    source = tmp_path / "source.onnx"
    source.write_bytes(b"ONNX graph")
    artifact = DeploymentArtifact(
        model=ModelFiles(format="onnx", path=source),
        inputs=(_spec("input_ids", DType.INT64, (1, 8), (1, 8)),),
        outputs=(_spec("logits", DType.FLOAT32, (1, 4), (1, 4)),),
        runtime=RuntimeConfig(name="onnxruntime", options={"execution_provider": "cuda"}),
    )

    model = aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")

    parsed = text_format.Parse((model / "config.pbtxt").read_text(), model_config_pb2.ModelConfig())
    assert parsed.max_batch_size == 1
    assert not parsed.HasField("dynamic_batching")
    assert tuple(parsed.input[0].dims) == (8,)
    assert tuple(parsed.output[0].dims) == (4,)


def test_publish_uses_smallest_batch_bound_across_inputs_and_outputs(tmp_path):
    artifact = _plan(tmp_path / "source.plan")
    artifact = replace(artifact, outputs=(_spec("logits", DType.FLOAT32, (1, 4), (4, 4)),))

    model = aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")

    parsed = text_format.Parse((model / "config.pbtxt").read_text(), model_config_pb2.ModelConfig())
    assert parsed.max_batch_size == 4
    assert parsed.HasField("dynamic_batching")


def test_publish_keeps_nonleading_batch_axis_in_full_shape(tmp_path):
    source = tmp_path / "source.onnx"
    source.write_bytes(b"ONNX graph")
    artifact = DeploymentArtifact(
        model=ModelFiles(format="onnx", path=source),
        inputs=(
            BoundedTensorSpec(name="images", dtype=DType.FLOAT32, min_shape=(3, 1), max_shape=(3, 8), batch_axis=1),
        ),
        outputs=(_spec("logits", DType.FLOAT32, (1, 4), (8, 4)),),
        runtime=RuntimeConfig(name="onnxruntime", options={"execution_provider": "cuda"}),
    )

    model = aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")

    parsed = text_format.Parse((model / "config.pbtxt").read_text(), model_config_pb2.ModelConfig())
    assert parsed.max_batch_size == 0
    assert not parsed.HasField("dynamic_batching")
    assert tuple(parsed.input[0].dims) == (3, -1)


@pytest.mark.parametrize(
    ("minimum", "maximum", "batch_axis", "dynamic_batching"),
    [
        ((), (), None, False),
        ((1,), (8,), 0, True),
    ],
)
def test_publishes_scalar_tensors_with_triton_reshape(tmp_path, minimum, maximum, batch_axis, dynamic_batching):
    artifact = _plan(tmp_path / "source.plan")
    scalar = BoundedTensorSpec(
        name="scalar",
        dtype=DType.FLOAT32,
        min_shape=minimum,
        max_shape=maximum,
        batch_axis=batch_axis,
    )
    artifact = replace(artifact, inputs=(scalar,), outputs=(scalar,))

    model = aitriton.publish(artifact, path=tmp_path / "repository", model_name="scalar")

    parsed = text_format.Parse((model / "config.pbtxt").read_text(), model_config_pb2.ModelConfig())
    assert parsed.HasField("dynamic_batching") == dynamic_batching
    for tensor in (parsed.input[0], parsed.output[0]):
        assert tuple(tensor.dims) == (1,)
        assert tensor.HasField("reshape")
        assert tuple(tensor.reshape.shape) == ()


@pytest.mark.parametrize(
    ("provider", "expected_accelerators"),
    [
        ("cuda", ()),
        ("tensorrt", ("tensorrt",)),
    ],
)
def test_publishes_onnx_external_data_and_runtime_provider(tmp_path, provider, expected_accelerators):
    source = tmp_path / "source"
    source.mkdir()
    model_file = source / "encoder.onnx"
    model_file.write_bytes(b"ONNX graph")
    weights = source / "weights.bin"
    weights.write_bytes(b"external weights")
    inputs, outputs = _interface()
    artifact = DeploymentArtifact(
        model=ModelFiles(format="onnx", path=model_file, additional_files=(Path("weights.bin"),)),
        inputs=inputs,
        outputs=outputs,
        runtime=RuntimeConfig(name="onnxruntime", options={"execution_provider": provider}),
    )

    published = aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")

    artifact_directory = published / "1" / "model.onnx"
    assert (artifact_directory / "model.onnx").read_bytes() == b"ONNX graph"
    assert (artifact_directory / "weights.bin").read_bytes() == b"external weights"
    config = (published / "config.pbtxt").read_text()
    parsed = text_format.Parse(config, model_config_pb2.ModelConfig())
    assert 'platform: "onnxruntime_onnx"' in config
    assert parsed.max_batch_size == 8
    assert parsed.HasField("dynamic_batching")
    assert tuple(parsed.input[0].dims) == (-1,)
    accelerators = parsed.optimization.execution_accelerators.gpu_execution_accelerator
    assert tuple(accelerator.name for accelerator in accelerators) == expected_accelerators


def test_publishes_pt2_using_torch_aoti_names(tmp_path):
    package = tmp_path / "source.pt2"
    package.write_bytes(b"PT2 package")
    artifact = DeploymentArtifact(
        model=ModelFiles(format="pt2", path=package, metadata={"structured_call": False}),
        inputs=(_spec("INPUT__0", DType.FLOAT32, (1, 16), (8, 16)),),
        outputs=(_spec("OUTPUT__0", DType.FLOAT32, (1, 8), (8, 8)),),
        runtime=RuntimeConfig(name="aotinductor"),
    )

    published = aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")

    assert (published / "1" / "model.pt2").read_bytes() == b"PT2 package"
    config = (published / "config.pbtxt").read_text()
    assert 'platform: "torch_aoti"' in config
    assert 'name: "INPUT__0"' in config
    assert 'name: "OUTPUT__0"' in config
    assert "dynamic_batching" in config


def test_structured_pt2_publishes_without_batching(tmp_path):
    package = tmp_path / "source.pt2"
    package.write_bytes(b"PT2 package")
    inputs, outputs = _interface()
    artifact = DeploymentArtifact(
        model=ModelFiles(format="pt2", path=package, metadata={"structured_call": True}),
        inputs=inputs,
        outputs=outputs,
        runtime=RuntimeConfig(name="aotinductor"),
    )

    published = aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")
    assert (published / "1" / "model.pt2").is_file()
    parsed = text_format.Parse((published / "config.pbtxt").read_text(), model_config_pb2.ModelConfig())
    assert parsed.max_batch_size == 0
    assert not parsed.HasField("dynamic_batching")


def test_refuses_to_replace_an_existing_model(tmp_path):
    artifact = _plan(tmp_path / "source.plan")
    repository = tmp_path / "repository"
    published = aitriton.publish(artifact, path=repository, model_name="encoder")
    original_config = (published / "config.pbtxt").read_text()

    with pytest.raises(AITunePublicationError, match="never replaces"):
        aitriton.publish(artifact, path=repository, model_name="encoder")

    assert (published / "config.pbtxt").read_text() == original_config


def test_repository_creation_failure_raises_publication_error(tmp_path):
    artifact = _plan(tmp_path / "source.plan")
    repository = tmp_path / "repository"
    repository.write_text("not a directory")

    with pytest.raises(AITunePublicationError, match="Failed to publish") as error:
        aitriton.publish(artifact, path=repository, model_name="encoder")

    assert isinstance(error.value.__cause__, OSError)


def test_publishes_to_current_directory(tmp_path, monkeypatch):
    artifact = _plan(tmp_path / "source.plan")
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.chdir(repository)

    aitriton.publish(artifact, path=".", model_name="encoder")

    assert (repository / "encoder" / "1" / "model.plan").read_bytes() == b"TensorRT plan"


def test_copy_failure_leaves_partial_model_for_caller_to_remove(tmp_path):
    artifact = _plan(tmp_path / "source.plan")
    artifact = replace(
        artifact,
        model=ModelFiles(format="onnx", path=artifact.model.path, additional_files=(Path("missing.bin"),)),
        runtime=RuntimeConfig(name="onnxruntime", options={"execution_provider": "cuda"}),
    )

    with pytest.raises(AITunePublicationError, match="An incomplete model directory may remain"):
        aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")

    assert (tmp_path / "repository" / "encoder" / "config.pbtxt").is_file()


@pytest.mark.parametrize(
    ("model_format", "runtime"),
    [("custom", "tensorrt"), ("tensorrt_plan", "custom"), ("onnx", "tensorrt"), ("pt2", "onnxruntime")],
)
def test_rejects_unsupported_format_runtime_pairs(tmp_path, model_format, runtime):
    artifact = _plan(tmp_path / "source.plan")
    artifact = replace(
        artifact,
        model=replace(artifact.model, format=model_format),
        runtime=RuntimeConfig(name=runtime),
    )

    with pytest.raises(AITunePublicationError, match="Unsupported Triton model format/runtime pair"):
        aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")

    assert not (tmp_path / "repository").exists()


@pytest.mark.parametrize("count", [None, 0, -1, True, "2"])
def test_rejects_invalid_tensorrt_profile_count(tmp_path, count):
    artifact = _plan(tmp_path / "source.plan")
    artifact = replace(artifact, model=replace(artifact.model, metadata={"optimization_profile_count": count}))

    with pytest.raises(AITunePublicationError, match="optimization_profile_count"):
        aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")

    assert not (tmp_path / "repository").exists()


@pytest.mark.parametrize(("model_format", "runtime"), [("tensorrt_plan", "tensorrt"), ("pt2", "aotinductor")])
def test_rejects_additional_files_for_single_file_formats(tmp_path, model_format, runtime):
    artifact = _plan(tmp_path / "source.plan")
    artifact = replace(
        artifact,
        model=replace(artifact.model, format=model_format, additional_files=(Path("weights.bin"),)),
        runtime=RuntimeConfig(name=runtime),
    )

    with pytest.raises(AITunePublicationError, match="artifact has additional files"):
        aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")

    assert not (tmp_path / "repository").exists()


@pytest.mark.parametrize("options", [{}, {"execution_provider": "unsupported"}])
def test_rejects_invalid_onnx_runtime_options(tmp_path, options):
    artifact = _plan(tmp_path / "source.plan")
    artifact = replace(
        artifact,
        model=ModelFiles(format="onnx", path=artifact.model.path),
        runtime=RuntimeConfig(name="onnxruntime", options=options),
    )

    with pytest.raises(AITunePublicationError, match="execution_provider"):
        aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")

    assert not (tmp_path / "repository").exists()


def test_requires_pt2_call_metadata(tmp_path):
    inputs, outputs = _interface()
    artifact = DeploymentArtifact(
        model=ModelFiles(format="pt2", path=tmp_path / "source.pt2"),
        inputs=inputs,
        outputs=outputs,
        runtime=RuntimeConfig(name="aotinductor"),
    )

    with pytest.raises(AITunePublicationError, match="structured_call"):
        aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")

    assert not (tmp_path / "repository").exists()
