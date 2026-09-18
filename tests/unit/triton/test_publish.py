# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from pathlib import Path

import pytest
from google.protobuf import text_format
from tritonclient.grpc import model_config_pb2

from aitune import triton as aitriton
from aitune.records import (
    BoundedTensorSpec,
    DeploymentArtifact,
    DType,
    ModelFiles,
    RuntimeConfig,
)


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
        model=ModelFiles(format="tensorrt_plan", path=path, metadata={"optimization_profile_count": profiles}),
        inputs=inputs,
        outputs=outputs,
        runtime=RuntimeConfig(name="tensorrt", options={"use_cuda_graphs": use_cuda_graphs}),
    )


def test_publishes_tensorrt_plan_with_bounds_batching_and_profiles(tmp_path):
    artifact = _plan(tmp_path / "source.plan", profiles=2, use_cuda_graphs=True)

    model = aitriton.publish(
        artifact,
        path=tmp_path / "repository",
        model_name="encoder",
        dynamic_batching=True,
        max_batch_size=4,
    )

    assert (model / "1" / "model.plan").read_bytes() == b"TensorRT plan"
    config = (model / "config.pbtxt").read_text()
    parsed = text_format.Parse(config, model_config_pb2.ModelConfig())
    assert parsed.name == "encoder"
    assert parsed.max_batch_size == 4
    assert 'platform: "tensorrt_plan"' in config
    assert "max_batch_size: 4" in config
    assert parsed.input[0].name == "input_ids"
    assert parsed.input[0].data_type == model_config_pb2.TYPE_INT64
    assert tuple(parsed.input[0].dims) == (-1,)
    assert tuple(parsed.instance_group[0].profile) == ("0", "1")
    assert parsed.optimization.cuda.graphs
    assert parsed.HasField("dynamic_batching")


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

    model = aitriton.publish(
        artifact,
        path=tmp_path / "repository",
        model_name="scalar",
        dynamic_batching=dynamic_batching,
    )

    parsed = text_format.Parse((model / "config.pbtxt").read_text(), model_config_pb2.ModelConfig())
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
    assert parsed.max_batch_size == 0
    assert tuple(parsed.input[0].dims) == (-1, -1)
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


def test_structured_pt2_publishes_unbatched_but_refuses_dynamic_batching(tmp_path):
    package = tmp_path / "source.pt2"
    package.write_bytes(b"PT2 package")
    inputs, outputs = _interface()
    artifact = DeploymentArtifact(
        model=ModelFiles(format="pt2", path=package, metadata={"structured_call": True}),
        inputs=inputs,
        outputs=outputs,
        runtime=RuntimeConfig(name="aotinductor"),
    )

    published = aitriton.publish(artifact, path=tmp_path / "unbatched", model_name="encoder")
    assert (published / "1" / "model.pt2").is_file()

    with pytest.raises(aitriton.AITunePublicationError, match="structured calls"):
        aitriton.publish(
            artifact,
            path=tmp_path / "repository",
            model_name="encoder",
            dynamic_batching=True,
        )

    assert not (tmp_path / "repository" / "encoder").exists()


def test_refuses_to_replace_an_existing_model(tmp_path):
    artifact = _plan(tmp_path / "source.plan")
    repository = tmp_path / "repository"
    published = aitriton.publish(artifact, path=repository, model_name="encoder")
    original_config = (published / "config.pbtxt").read_text()

    with pytest.raises(aitriton.AITunePublicationError, match="never replaces"):
        aitriton.publish(artifact, path=repository, model_name="encoder")

    assert (published / "config.pbtxt").read_text() == original_config


def test_repository_creation_failure_raises_publication_error(tmp_path):
    artifact = _plan(tmp_path / "source.plan")
    repository = tmp_path / "repository"
    repository.write_text("not a directory")

    with pytest.raises(aitriton.AITunePublicationError, match="Failed to publish") as error:
        aitriton.publish(artifact, path=repository, model_name="encoder")

    assert isinstance(error.value.__cause__, OSError)


def test_staging_directory_creation_failure_raises_publication_error(tmp_path, mocker):
    artifact = _plan(tmp_path / "source.plan")
    repository = tmp_path / "repository"
    make_staging = mocker.patch(
        "aitune.triton.model_repository.tempfile.mkdtemp", side_effect=OSError("staging unavailable")
    )

    with pytest.raises(aitriton.AITunePublicationError, match="staging unavailable") as error:
        aitriton.publish(artifact, path=repository, model_name="encoder")

    assert isinstance(error.value.__cause__, OSError)
    make_staging.assert_called_once_with(dir=tmp_path)
    assert not (repository / "encoder").exists()


def test_uses_explicit_staging_path_outside_repository(tmp_path):
    artifact = _plan(tmp_path / "source.plan")
    repository = tmp_path / "repository"
    staging_root = tmp_path / "staging"

    published = aitriton.publish(
        artifact,
        path=repository,
        model_name="encoder",
        staging_path=staging_root,
    )

    assert published == repository / "encoder"
    assert not tuple(staging_root.iterdir())


def test_rejects_staging_path_inside_repository_before_export(tmp_path, mocker):
    artifact = _plan(tmp_path / "source.plan")
    repository = tmp_path / "repository"
    export_files = mocker.patch.object(ModelFiles, "export_files")

    with pytest.raises(aitriton.AITunePublicationError, match="outside the model repository"):
        aitriton.publish(
            artifact,
            path=repository,
            model_name="encoder",
            staging_path=repository / "staging",
        )

    export_files.assert_not_called()


def test_rejects_staging_path_on_different_filesystem_before_export(tmp_path, mocker):
    artifact = _plan(tmp_path / "source.plan")
    export_files = mocker.patch.object(ModelFiles, "export_files")
    mocker.patch("aitune.triton.model_repository._same_filesystem", return_value=False)

    with pytest.raises(aitriton.AITunePublicationError, match="same filesystem"):
        aitriton.publish(
            artifact,
            path=tmp_path / "repository",
            model_name="encoder",
            staging_path=tmp_path / "staging",
        )

    export_files.assert_not_called()


def test_copy_failure_leaves_no_partial_model(tmp_path):
    artifact = _plan(tmp_path / "source.plan")
    artifact = replace(
        artifact,
        model=ModelFiles(format="onnx", path=artifact.model.path, additional_files=(Path("missing.bin"),)),
        runtime=RuntimeConfig(name="onnxruntime", options={"execution_provider": "cuda"}),
    )

    with pytest.raises(aitriton.AITunePublicationError, match="Failed to publish"):
        aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")

    assert not tuple((tmp_path / "repository").iterdir())


def test_cleanup_failure_is_reported(tmp_path, mocker):
    artifact = _plan(tmp_path / "source.plan")
    artifact = replace(
        artifact,
        model=ModelFiles(format="onnx", path=artifact.model.path, additional_files=(Path("missing.bin"),)),
        runtime=RuntimeConfig(name="onnxruntime", options={"execution_provider": "cuda"}),
    )
    mocker.patch("aitune.triton.model_repository.shutil.rmtree", side_effect=OSError("cleanup unavailable"))

    with pytest.raises(aitriton.AITunePublicationError, match="failed to clean staging directory") as error:
        aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")

    assert isinstance(error.value.__cause__, OSError)
    assert "cleanup unavailable" in str(error.value)


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

    with pytest.raises(aitriton.AITunePublicationError, match="Unsupported Triton model format/runtime pair"):
        aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")

    assert not (tmp_path / "repository").exists()


@pytest.mark.parametrize("count", [None, 0, -1, True, "2"])
def test_rejects_invalid_tensorrt_profile_count(tmp_path, count):
    artifact = _plan(tmp_path / "source.plan")
    artifact = replace(artifact, model=replace(artifact.model, metadata={"optimization_profile_count": count}))

    with pytest.raises(aitriton.AITunePublicationError, match="optimization_profile_count"):
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

    with pytest.raises(aitriton.AITunePublicationError, match="artifact has additional files"):
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

    with pytest.raises(aitriton.AITunePublicationError, match="execution_provider"):
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

    with pytest.raises(aitriton.AITunePublicationError, match="structured_call"):
        aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")

    assert not (tmp_path / "repository").exists()
