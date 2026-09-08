# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path

import pytest

from aitune.records import (
    ArtifactFile,
    BoundedTensorSpec,
    DType,
    ONNXArtifact,
    ONNXExecutionProvider,
    PT2Artifact,
    TensorRTPlanArtifact,
)
from aitune.records.artifact import ArtifactIntegrityError

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


def _write_artifact(tmp_path, contents=b"artifact-bytes") -> tuple[Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "model.bin"
    path.write_bytes(contents)
    return path, hashlib.sha256(contents).hexdigest()


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


def test_artifact_preserves_tensor_order_and_shared_batch_limit(tmp_path):
    path, fingerprint = _write_artifact(tmp_path, b"onnx")
    second_input = BoundedTensorSpec(
        name="mask",
        dtype=DType.BOOL,
        min_shape=(1, 8),
        max_shape=(4, 512),
        batch_axis=0,
    )

    artifact = ONNXArtifact(
        inputs=(*INPUTS, second_input),
        outputs=OUTPUTS,
        path=path,
        fingerprint=fingerprint,
    )

    assert artifact.input_names == ("input_ids", "mask")
    assert artifact.output_names == ("embedding",)
    assert artifact.max_batch_size == 4


def test_artifact_without_a_shared_batch_axis_has_no_batch_limit(tmp_path):
    path, fingerprint = _write_artifact(tmp_path)
    output = BoundedTensorSpec(
        name="score",
        dtype=DType.FLOAT32,
        min_shape=(1,),
        max_shape=(1,),
        batch_axis=None,
    )

    artifact = ONNXArtifact(inputs=INPUTS, outputs=(output,), path=path, fingerprint=fingerprint)

    assert artifact.max_batch_size is None


def test_artifact_without_batch_size_one_has_no_batch_limit(tmp_path):
    path, fingerprint = _write_artifact(tmp_path)
    input_spec = BoundedTensorSpec(
        name="input_ids",
        dtype=DType.INT64,
        min_shape=(2, 8),
        max_shape=(8, 512),
        batch_axis=0,
    )

    artifact = ONNXArtifact(inputs=(input_spec,), outputs=OUTPUTS, path=path, fingerprint=fingerprint)

    assert artifact.max_batch_size is None


def test_artifact_rejects_duplicate_tensor_names(tmp_path):
    path, fingerprint = _write_artifact(tmp_path)

    with pytest.raises(ValueError, match="input tensor names must be unique"):
        ONNXArtifact(inputs=(INPUTS[0], INPUTS[0]), outputs=OUTPUTS, path=path, fingerprint=fingerprint)


def test_artifact_formats_preserve_runtime_requirements(tmp_path):
    path, fingerprint = _write_artifact(tmp_path)

    plan = TensorRTPlanArtifact(
        inputs=INPUTS,
        outputs=OUTPUTS,
        path=path,
        fingerprint=fingerprint,
        optimization_profile_count=2,
        use_cuda_graphs=True,
    )
    onnx = ONNXArtifact(
        inputs=INPUTS,
        outputs=OUTPUTS,
        path=path,
        fingerprint=fingerprint,
        execution_provider=ONNXExecutionProvider.TENSORRT,
    )
    pt2 = PT2Artifact(
        inputs=INPUTS,
        outputs=OUTPUTS,
        path=path,
        fingerprint=fingerprint,
        structured_call=True,
    )

    assert (plan.optimization_profile_count, plan.use_cuda_graphs) == (2, True)
    assert onnx.execution_provider is ONNXExecutionProvider.TENSORRT
    assert pt2.structured_call is True


def test_artifact_refuses_changed_or_missing_bytes(tmp_path):
    path, fingerprint = _write_artifact(tmp_path, b"plan")
    artifact = TensorRTPlanArtifact(inputs=INPUTS, outputs=OUTPUTS, path=path, fingerprint=fingerprint)
    artifact.verify()

    path.write_bytes(b"changed")
    with pytest.raises(ArtifactIntegrityError, match="has changed since it was built"):
        artifact.verify()

    path.unlink()
    with pytest.raises(ArtifactIntegrityError, match="cache may have been cleared"):
        artifact.verify()


def test_export_file_copies_only_verified_bytes(tmp_path):
    path, fingerprint = _write_artifact(tmp_path / "cache", b"plan")
    artifact = TensorRTPlanArtifact(inputs=INPUTS, outputs=OUTPUTS, path=path, fingerprint=fingerprint)
    destination = tmp_path / "repository" / "model.plan"

    assert artifact.export_file(destination).read_bytes() == b"plan"

    path.write_bytes(b"changed")
    with pytest.raises(ArtifactIntegrityError):
        artifact.export_file(destination)
    assert destination.read_bytes() == b"plan"


def test_artifact_exports_verified_companion_files(tmp_path):
    path, fingerprint = _write_artifact(tmp_path / "cache", b"onnx")
    weights = path.parent / "weights" / "model.data"
    weights.parent.mkdir()
    weights.write_bytes(b"weights")
    weights_fingerprint = hashlib.sha256(b"weights").hexdigest()
    artifact = ONNXArtifact(
        inputs=INPUTS,
        outputs=OUTPUTS,
        path=path,
        fingerprint=fingerprint,
        companions=(ArtifactFile(Path("weights/model.data"), weights_fingerprint),),
    )

    destination = tmp_path / "repository" / "model.onnx"
    artifact.export_file(destination)

    assert destination.read_bytes() == b"onnx"
    assert (destination.parent / "weights" / "model.data").read_bytes() == b"weights"


def test_changed_companion_does_not_replace_existing_export(tmp_path):
    path, fingerprint = _write_artifact(tmp_path / "cache", b"new-model")
    weights = path.parent / "model.data"
    weights.write_bytes(b"expected-weights")
    artifact = ONNXArtifact(
        inputs=INPUTS,
        outputs=OUTPUTS,
        path=path,
        fingerprint=fingerprint,
        companions=(ArtifactFile(Path("model.data"), hashlib.sha256(b"expected-weights").hexdigest()),),
    )
    destination = tmp_path / "repository" / "model.onnx"
    destination.parent.mkdir()
    destination.write_bytes(b"old-model")
    weights.write_bytes(b"changed-weights")

    with pytest.raises(ArtifactIntegrityError):
        artifact.export_file(destination)

    assert destination.read_bytes() == b"old-model"


@pytest.mark.parametrize("relative_path", [Path(), Path("../model.data"), Path("/model.data")])
def test_artifact_companion_must_stay_inside_artifact_directory(relative_path):
    with pytest.raises(ValueError, match="must stay inside"):
        ArtifactFile(relative_path, "0" * 64)


def test_artifact_requires_a_sha256_fingerprint(tmp_path):
    path, _ = _write_artifact(tmp_path)

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        ONNXArtifact(inputs=INPUTS, outputs=OUTPUTS, path=path, fingerprint="not-a-digest")
