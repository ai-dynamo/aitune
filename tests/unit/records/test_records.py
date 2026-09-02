# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path

import pytest

from aitune.records import DType, ONNXArtifact, TensorRTPlanArtifact, TensorSpec, TunedTensorSpec
from aitune.records.artifact import ArtifactIntegrityError

INPUT_SPEC = TensorSpec("input_ids", DType.INT64, ("batch", "sequence"))
OUTPUT_SPEC = TensorSpec("embedding", DType.FLOAT32, ("batch", "sequence", 768))
INPUTS = (TunedTensorSpec.from_spec(INPUT_SPEC, min_shape=(1, 8), max_shape=(8, 512), batch_axis=0),)
OUTPUTS = (TunedTensorSpec.from_spec(OUTPUT_SPEC, min_shape=(1, 8, 768), max_shape=(8, 512, 768), batch_axis=0),)


def _write_artifact(tmp_path, contents=b"artifact-bytes") -> tuple[Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "model.bin"
    path.write_bytes(contents)
    return Path(path), hashlib.sha256(contents).hexdigest()


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


def test_tensor_spec_describes_static_symbolic_and_unknown_dimensions():
    spec = TensorSpec("value", DType.FLOAT32, (1, "sequence", None))

    assert spec.shape == (1, "sequence", None)


@pytest.mark.parametrize("name", ["", 1])
def test_tensor_spec_rejects_an_invalid_name(name):
    with pytest.raises(ValueError, match=r"TensorSpec\.name must be a non-empty string"):
        TensorSpec(name, DType.FLOAT32, (1,))


def test_tensor_spec_rejects_a_non_tuple_shape():
    with pytest.raises(ValueError, match=r"TensorSpec\.shape must be a tuple"):
        TensorSpec("input_ids", DType.INT64, "batch")  # pytype: disable=wrong-arg-types


@pytest.mark.parametrize("dimension", [True, 0, -1, "", 1.5])
def test_tensor_spec_rejects_an_invalid_dimension(dimension):
    with pytest.raises(ValueError, match=r"TensorSpec\.shape\[0\]"):
        TensorSpec("value", DType.FLOAT32, (dimension,))


def test_tuned_tensor_spec_combines_a_spec_with_tuned_bounds():
    metadata = TunedTensorSpec.from_spec(
        TensorSpec("value", DType.FLOAT32, ("batch", None, 4)),
        min_shape=(1, 8, 4),
        max_shape=(16, 512, 4),
        batch_axis=0,
    )

    assert isinstance(metadata, TensorSpec)
    assert metadata.name == "value"
    assert metadata.shape == ("batch", None, 4)
    assert metadata.min_batch_size == 1
    assert metadata.max_batch_size == 16


@pytest.mark.parametrize(
    ("min_shape", "max_shape"),
    [
        ((1,), (1, 2)),
        ((1, 2), (1,)),
    ],
)
def test_tuned_tensor_spec_requires_the_spec_rank(min_shape, max_shape):
    with pytest.raises(ValueError, match="must have rank 2"):
        TunedTensorSpec.from_spec(INPUT_SPEC, min_shape=min_shape, max_shape=max_shape)


@pytest.mark.parametrize(
    ("min_shape", "max_shape"),
    [
        ((0,), (1,)),
        ((True,), (1,)),
        ((2,), (1,)),
    ],
)
def test_tuned_tensor_spec_rejects_invalid_bounds(min_shape, max_shape):
    spec = TensorSpec("value", DType.FLOAT32, (None,))

    with pytest.raises(ValueError):
        TunedTensorSpec.from_spec(spec, min_shape=min_shape, max_shape=max_shape)


def test_tuned_tensor_spec_preserves_fixed_dimensions():
    spec = TensorSpec("value", DType.FLOAT32, (None, 4))

    with pytest.raises(ValueError, match="Fixed axis 1.*must remain 4"):
        TunedTensorSpec.from_spec(spec, min_shape=(1, 8), max_shape=(8, 8))


@pytest.mark.parametrize("batch_axis", [-1, 2, True, "0"])
def test_tuned_tensor_spec_requires_a_valid_batch_axis(batch_axis):
    with pytest.raises(ValueError, match="batch_axis must be a valid axis"):
        TunedTensorSpec.from_spec(INPUT_SPEC, min_shape=(1, 8), max_shape=(8, 512), batch_axis=batch_axis)


def test_tuned_tensor_spec_without_a_batch_axis_has_no_batch_size():
    metadata = TunedTensorSpec.from_spec(INPUT_SPEC, min_shape=(1, 8), max_shape=(8, 512))

    assert metadata.min_batch_size is None
    assert metadata.max_batch_size is None


def test_artifact_preserves_ordered_input_and_output_metadata(tmp_path):
    path, fingerprint = _write_artifact(tmp_path, b"onnx")
    inputs = (
        TunedTensorSpec.from_spec(
            TensorSpec("tokens", DType.INT64, ("batch", "sequence")),
            min_shape=(1, 8),
            max_shape=(8, 512),
            batch_axis=0,
        ),
        TunedTensorSpec.from_spec(
            TensorSpec("mask", DType.BOOL, ("batch", "sequence")),
            min_shape=(1, 8),
            max_shape=(8, 512),
            batch_axis=0,
        ),
    )
    outputs = (
        TunedTensorSpec.from_spec(
            TensorSpec("hidden", DType.FLOAT16, ("batch", "sequence", 768)),
            min_shape=(1, 8, 768),
            max_shape=(8, 512, 768),
            batch_axis=0,
        ),
        TunedTensorSpec.from_spec(
            TensorSpec("pooled", DType.FLOAT16, ("batch", 768)),
            min_shape=(1, 768),
            max_shape=(8, 768),
            batch_axis=0,
        ),
    )

    artifact = ONNXArtifact(inputs=inputs, outputs=outputs, path=path, fingerprint=fingerprint)

    assert artifact.inputs == inputs
    assert artifact.outputs == outputs
    assert artifact.input_names == ("tokens", "mask")
    assert artifact.output_names == ("hidden", "pooled")


def test_artifact_rejects_duplicate_names_in_each_section(tmp_path):
    path, fingerprint = _write_artifact(tmp_path)

    with pytest.raises(ValueError, match="input tensor names must be unique"):
        ONNXArtifact(inputs=(INPUTS[0], INPUTS[0]), outputs=(), path=path, fingerprint=fingerprint)

    with pytest.raises(ValueError, match="output tensor names must be unique"):
        ONNXArtifact(inputs=(), outputs=(OUTPUTS[0], OUTPUTS[0]), path=path, fingerprint=fingerprint)


def test_artifact_allows_the_same_name_across_inputs_and_outputs(tmp_path):
    path, fingerprint = _write_artifact(tmp_path)
    spec = TensorSpec("value", DType.FLOAT32, (1,))
    metadata = TunedTensorSpec.from_spec(spec, min_shape=(1,), max_shape=(1,))

    artifact = ONNXArtifact(inputs=(metadata,), outputs=(metadata,), path=path, fingerprint=fingerprint)

    assert artifact.input_names == artifact.output_names == ("value",)


@pytest.mark.parametrize("artifact_type", [ONNXArtifact, TensorRTPlanArtifact])
def test_artifact_returns_the_shared_maximum_batch_size(tmp_path, artifact_type):
    path, fingerprint = _write_artifact(tmp_path)

    artifact = artifact_type(inputs=INPUTS, outputs=OUTPUTS, path=path, fingerprint=fingerprint)

    assert artifact.max_batch_size == 8


def test_artifact_without_a_batch_axis_has_no_maximum_batch_size(tmp_path):
    path, fingerprint = _write_artifact(tmp_path)
    metadata = TunedTensorSpec.from_spec(TensorSpec("value", DType.FLOAT32, (4,)), min_shape=(4,), max_shape=(4,))

    artifact = ONNXArtifact(inputs=(metadata,), outputs=(), path=path, fingerprint=fingerprint)

    assert artifact.max_batch_size is None


def test_artifact_intersects_different_tensor_maximum_batch_sizes(tmp_path):
    path, fingerprint = _write_artifact(tmp_path)
    other = TunedTensorSpec.from_spec(
        TensorSpec("mask", DType.BOOL, ("batch", "sequence")),
        min_shape=(1, 8),
        max_shape=(4, 512),
        batch_axis=0,
    )

    artifact = ONNXArtifact(inputs=(*INPUTS, other), outputs=OUTPUTS, path=path, fingerprint=fingerprint)

    assert artifact.max_batch_size == 4


def test_artifact_requires_every_tensor_to_have_a_batch_axis(tmp_path):
    path, fingerprint = _write_artifact(tmp_path)
    output = TunedTensorSpec.from_spec(
        TensorSpec("score", DType.FLOAT32, (1,)),
        min_shape=(1,),
        max_shape=(1,),
    )

    artifact = ONNXArtifact(inputs=INPUTS, outputs=(output,), path=path, fingerprint=fingerprint)

    assert artifact.max_batch_size is None


def test_artifact_requires_support_for_batch_size_one(tmp_path):
    path, fingerprint = _write_artifact(tmp_path)
    input_spec = TunedTensorSpec.from_spec(
        INPUT_SPEC,
        min_shape=(2, 8),
        max_shape=(8, 512),
        batch_axis=0,
    )

    artifact = ONNXArtifact(inputs=(input_spec,), outputs=OUTPUTS, path=path, fingerprint=fingerprint)

    assert artifact.max_batch_size is None


def test_artifact_verifies_unchanged_bytes(tmp_path):
    path, fingerprint = _write_artifact(tmp_path, b"plan")
    artifact = TensorRTPlanArtifact(inputs=INPUTS, outputs=OUTPUTS, path=path, fingerprint=fingerprint)

    artifact.verify()


def test_artifact_refuses_changed_or_missing_bytes(tmp_path):
    path, fingerprint = _write_artifact(tmp_path, b"plan")
    artifact = TensorRTPlanArtifact(inputs=INPUTS, outputs=OUTPUTS, path=path, fingerprint=fingerprint)
    path.write_bytes(b"changed")

    with pytest.raises(ArtifactIntegrityError, match="has changed since it was built"):
        artifact.verify()

    path.unlink()
    with pytest.raises(ArtifactIntegrityError, match="cache may have been cleared"):
        artifact.verify()


def test_export_file_copies_only_verified_bytes(tmp_path):
    path, fingerprint = _write_artifact(tmp_path / "cache", b"plan")
    artifact = TensorRTPlanArtifact(inputs=INPUTS, outputs=OUTPUTS, path=path, fingerprint=fingerprint)
    destination_path = tmp_path / "repository" / "model.plan"
    destination_path.parent.mkdir()
    destination_path.write_bytes(b"old-plan")

    destination = artifact.export_file(destination_path)

    assert destination.read_bytes() == b"plan"

    changed_destination = tmp_path / "repository" / "changed.plan"
    changed_destination.write_bytes(b"existing-plan")
    path.write_bytes(b"changed")
    with pytest.raises(ArtifactIntegrityError):
        artifact.export_file(changed_destination)
    assert changed_destination.read_bytes() == b"existing-plan"
    assert sorted(item.name for item in destination_path.parent.iterdir()) == ["changed.plan", "model.plan"]


def test_export_file_preserves_destination_when_replace_fails(tmp_path, monkeypatch):
    path, fingerprint = _write_artifact(tmp_path / "cache", b"plan")
    artifact = TensorRTPlanArtifact(inputs=INPUTS, outputs=OUTPUTS, path=path, fingerprint=fingerprint)
    destination = tmp_path / "repository" / "model.plan"
    destination.parent.mkdir()
    destination.write_bytes(b"existing-plan")

    def fail_replace(_source, _target):
        raise OSError("replace failed")

    monkeypatch.setattr("aitune.records.artifact.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        artifact.export_file(destination)
    assert destination.read_bytes() == b"existing-plan"
    assert list(destination.parent.iterdir()) == [destination]


def test_artifact_requires_a_sha256_fingerprint(tmp_path):
    path, _ = _write_artifact(tmp_path)

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        ONNXArtifact(inputs=INPUTS, outputs=OUTPUTS, path=path, fingerprint="not-a-digest")
