# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from aitune.records import (
    BoundedTensorSpec,
    DeploymentArtifact,
    DType,
    ModelFiles,
    RuntimeConfig,
    TensorSample,
)

INPUTS = (
    BoundedTensorSpec(
        name="input_ids",
        dtype=DType.INT64,
        min_shape=(1, 8),
        max_shape=(8, 512),
        batch_axis=0,
    ),
)
OUTPUTS = (
    BoundedTensorSpec(
        name="embedding",
        dtype=DType.FLOAT32,
        min_shape=(1, 8, 768),
        max_shape=(8, 512, 768),
        batch_axis=0,
    ),
)


def _write_artifact(tmp_path, contents=b"artifact-bytes") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "model.bin"
    path.write_bytes(contents)
    return path


def test_dtype_vocabulary_is_stable_and_frontend_neutral():
    assert tuple(dtype.value for dtype in DType) == (
        "bool",
        "uint8",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
    )


def test_bounded_tensor_spec_describes_the_validated_tensor_domain():
    spec = BoundedTensorSpec(
        name="value",
        dtype=DType.FLOAT32,
        min_shape=(1, 8, 4),
        max_shape=(16, 512, 4),
        batch_axis=0,
    )

    assert spec.min_shape == (1, 8, 4)
    assert spec.max_shape == (16, 512, 4)
    assert spec.min_batch_size == 1
    assert spec.max_batch_size == 16


@pytest.mark.parametrize(
    ("min_shape", "max_shape"),
    [
        ((1,), (8, 4)),
        ((1, 4), (8,)),
        ((0,), (1,)),
        ((2,), (1,)),
    ],
)
def test_bounded_tensor_spec_rejects_inconsistent_bounds(min_shape, max_shape):
    with pytest.raises(ValueError):
        BoundedTensorSpec(
            name="value",
            dtype=DType.FLOAT32,
            min_shape=min_shape,
            max_shape=max_shape,
        )


@pytest.mark.parametrize("batch_axis", [-1, 2, True, "0"])
def test_bounded_tensor_spec_requires_a_valid_batch_axis(batch_axis):
    with pytest.raises(ValueError, match="batch_axis must be a valid axis"):
        BoundedTensorSpec(
            name="value",
            dtype=DType.FLOAT32,
            min_shape=(1, 4),
            max_shape=(8, 4),
            batch_axis=batch_axis,
        )


def test_tensor_sample_round_trips_checkpoint_values():
    sample = TensorSample(name="input_ids", shape=(2, 3), values=(1, 2, 3, 4, 5, 6))

    assert TensorSample.from_dict(sample.to_dict()) == sample


def test_tensor_sample_requires_values_matching_its_shape():
    with pytest.raises(ValueError, match="values do not match shape"):
        TensorSample(name="input_ids", shape=(2, 3), values=(1, 2, 3))


def test_artifact_preserves_tensor_order_and_shared_batch_limit(tmp_path):
    path = _write_artifact(tmp_path, b"onnx")
    second_input = BoundedTensorSpec(
        name="mask",
        dtype=DType.BOOL,
        min_shape=(1, 8),
        max_shape=(4, 512),
        batch_axis=0,
    )

    artifact = DeploymentArtifact(
        inputs=(*INPUTS, second_input),
        outputs=OUTPUTS,
        model=ModelFiles(format="onnx", path=path),
        runtime=RuntimeConfig(name="onnxruntime"),
    )

    assert artifact.input_names == ("input_ids", "mask")
    assert artifact.output_names == ("embedding",)
    assert artifact.max_batch_size == 4


def test_artifact_without_a_shared_batch_axis_has_no_batch_limit(tmp_path):
    path = _write_artifact(tmp_path)
    output = BoundedTensorSpec(
        name="score",
        dtype=DType.FLOAT32,
        min_shape=(1,),
        max_shape=(1,),
        batch_axis=None,
    )

    artifact = DeploymentArtifact(
        inputs=INPUTS,
        outputs=(output,),
        model=ModelFiles(format="onnx", path=path),
        runtime=RuntimeConfig(name="onnxruntime"),
    )

    assert artifact.max_batch_size is None


def test_artifact_without_batch_size_one_has_no_batch_limit(tmp_path):
    path = _write_artifact(tmp_path)
    input_spec = BoundedTensorSpec(
        name="input_ids",
        dtype=DType.INT64,
        min_shape=(2, 8),
        max_shape=(8, 512),
        batch_axis=0,
    )

    artifact = DeploymentArtifact(
        inputs=(input_spec,),
        outputs=OUTPUTS,
        model=ModelFiles(format="onnx", path=path),
        runtime=RuntimeConfig(name="onnxruntime"),
    )

    assert artifact.max_batch_size is None


def test_artifact_rejects_duplicate_tensor_names(tmp_path):
    path = _write_artifact(tmp_path)

    with pytest.raises(ValueError, match="input tensor names must be unique"):
        DeploymentArtifact(
            inputs=(INPUTS[0], INPUTS[0]),
            outputs=OUTPUTS,
            model=ModelFiles(format="onnx", path=path),
            runtime=RuntimeConfig(name="onnxruntime"),
        )


def test_artifact_requires_representative_values_in_input_order(tmp_path):
    path = _write_artifact(tmp_path)
    sample = TensorSample(name="mask", shape=(1, 8), values=(True,) * 8)

    with pytest.raises(ValueError, match="sample names must match artifact inputs"):
        DeploymentArtifact(
            inputs=INPUTS,
            outputs=OUTPUTS,
            model=ModelFiles(format="onnx", path=path),
            runtime=RuntimeConfig(name="onnxruntime"),
            sample_inputs=(sample,),
        )


@pytest.mark.parametrize(
    ("model_format", "metadata", "runtime_name", "options"),
    [
        ("tensorrt_plan", {"optimization_profile_count": 2}, "tensorrt", {"use_cuda_graphs": True}),
        ("onnx", {}, "onnxruntime", {"execution_provider": "tensorrt"}),
        ("pt2", {"structured_call": True}, "aotinductor", {}),
        ("custom_format", {"version": 1}, "custom_runtime", {"workers": 4}),
    ],
)
def test_deployment_artifact_preserves_format_metadata_and_runtime_options(
    tmp_path, model_format, metadata, runtime_name, options
):
    model = ModelFiles(format=model_format, path=_write_artifact(tmp_path), metadata=metadata)
    runtime = RuntimeConfig(name=runtime_name, options=options)

    artifact = DeploymentArtifact(model=model, inputs=INPUTS, outputs=OUTPUTS, runtime=runtime)

    assert artifact.model.format == model_format
    assert artifact.model.metadata == metadata
    assert artifact.runtime.name == runtime_name
    assert artifact.runtime.options == options


def test_same_model_files_can_be_used_with_different_runtimes(tmp_path):
    model = ModelFiles(format="onnx", path=_write_artifact(tmp_path))
    runtimes = (
        RuntimeConfig(name="onnxruntime", options={"execution_provider": "cuda"}),
        RuntimeConfig(name="custom_runtime", options={"threads": 2}),
    )
    artifacts = [
        DeploymentArtifact(model=model, inputs=INPUTS, outputs=OUTPUTS, runtime=runtime) for runtime in runtimes
    ]

    assert all(artifact.model is model for artifact in artifacts)
    assert tuple(artifact.runtime.name for artifact in artifacts) == ("onnxruntime", "custom_runtime")


def test_export_files_raises_for_missing_source(tmp_path):
    model_files = ModelFiles(format="tensorrt_plan", path=tmp_path / "missing.plan")

    with pytest.raises(FileNotFoundError):
        model_files.export_files(tmp_path / "repository" / "model.plan")


def test_export_files_rejects_main_file_renamed_to_additional_file(tmp_path):
    path = _write_artifact(tmp_path / "cache", b"model")
    weights = path.parent / "weights.data"
    weights.write_bytes(b"weights")
    model_files = ModelFiles(format="onnx", path=path, additional_files=(Path("weights.data"),))
    destination = tmp_path / "repository" / "weights.data"

    with pytest.raises(ValueError, match="exported to the same target"):
        model_files.export_files(destination)

    assert not destination.parent.exists()
    assert path.read_bytes() == b"model"
    assert weights.read_bytes() == b"weights"


def test_export_files_rejects_destination_matching_nested_additional_source(tmp_path):
    path = _write_artifact(tmp_path / "cache", b"model")
    weights = path.parent / "weights" / "model.data"
    weights.parent.mkdir()
    weights.write_bytes(b"weights")
    model_files = ModelFiles(format="onnx", path=path, additional_files=(Path("weights/model.data"),))

    with pytest.raises(ValueError, match="overwrite one of the model's source files"):
        model_files.export_files(weights)

    assert path.read_bytes() == b"model"
    assert weights.read_bytes() == b"weights"
    assert not (weights.parent / "weights" / "model.data").exists()


def test_export_files_copies_current_bytes(tmp_path):
    path = _write_artifact(tmp_path / "cache", b"plan")
    model_files = ModelFiles(format="tensorrt_plan", path=path)
    destination = tmp_path / "repository" / "model.plan"

    assert model_files.export_files(destination).read_bytes() == b"plan"

    path.write_bytes(b"changed")
    assert model_files.export_files(destination).read_bytes() == b"changed"


@pytest.mark.parametrize("destination_directory", ["repository", "cache"])
@pytest.mark.parametrize("destination_name", ["model.onnx", "model.bin"])
def test_artifact_exports_additional_files(tmp_path, destination_directory, destination_name):
    path = _write_artifact(tmp_path / "cache", b"onnx")
    weights = path.parent / "weights" / "model.data"
    weights.parent.mkdir()
    weights.write_bytes(b"weights")
    model_files = ModelFiles(
        format="onnx",
        path=path,
        additional_files=(Path("weights/model.data"),),
    )

    assert model_files.files == (path, weights)

    destination = tmp_path / destination_directory / destination_name
    assert model_files.export_files(destination) == destination

    assert destination.read_bytes() == b"onnx"
    assert (destination.parent / "weights" / "model.data").read_bytes() == b"weights"


def test_export_files_copies_current_additional_file_bytes(tmp_path):
    path = _write_artifact(tmp_path / "cache", b"new-model")
    weights = path.parent / "model.data"
    weights.write_bytes(b"expected-weights")
    model_files = ModelFiles(
        format="onnx",
        path=path,
        additional_files=(Path("model.data"),),
    )
    destination = tmp_path / "repository" / "model.onnx"
    destination.parent.mkdir()
    destination.write_bytes(b"old-model")
    weights.write_bytes(b"changed-weights")

    model_files.export_files(destination)

    assert destination.read_bytes() == b"new-model"
    assert (destination.parent / "model.data").read_bytes() == b"changed-weights"


def test_export_files_rejects_additional_file_symlink_outside_model_directory(tmp_path):
    path = _write_artifact(tmp_path / "cache", b"model")
    outside = tmp_path / "outside.data"
    outside.write_bytes(b"outside")
    link = path.parent / "weights.data"
    link.symlink_to(outside)
    model_files = ModelFiles(format="onnx", path=path, additional_files=(Path("weights.data"),))
    destination = tmp_path / "repository" / "model.onnx"

    with pytest.raises(ValueError, match="must stay inside the model directory"):
        model_files.export_files(destination)

    assert not destination.parent.exists()


@pytest.mark.parametrize("relative_path", [Path(), Path("../model.data"), Path("/model.data")])
def test_artifact_additional_file_must_stay_inside_artifact_directory(relative_path):
    with pytest.raises(ValueError, match="must stay inside"):
        ModelFiles(format="onnx", path=Path("model.onnx"), additional_files=(relative_path,))


@pytest.mark.parametrize(
    ("additional_files", "message"),
    [
        ((Path("weights.data"), Path("weights.data")), "must be unique"),
        ((Path("model.onnx"),), "cannot also be the model's main file"),
    ],
)
def test_model_files_reject_conflicting_paths(additional_files, message):
    with pytest.raises(ValueError, match=message):
        ModelFiles(format="onnx", path=Path("model.onnx"), additional_files=additional_files)
