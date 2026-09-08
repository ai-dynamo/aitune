# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path

import pytest
from google.protobuf import text_format
from tritonclient.grpc import model_config_pb2

from aitune import triton as aitriton
from aitune.records import (
    ArtifactFile,
    BoundedTensorSpec,
    DType,
    ONNXArtifact,
    ONNXExecutionProvider,
    PT2Artifact,
    TensorRTOptimizationProfile,
    TensorRTPlanArtifact,
    TensorRTProfileInput,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _plan(path: Path, *, profiles: int = 1, use_cuda_graphs: bool = False) -> TensorRTPlanArtifact:
    path.write_bytes(b"TensorRT plan")
    inputs, outputs = _interface()
    profile = TensorRTOptimizationProfile(
        inputs=(
            TensorRTProfileInput(
                name="input_ids",
                min_shape=(1, 8),
                opt_shape=(4, 128),
                max_shape=(8, 512),
            ),
        )
    )
    return TensorRTPlanArtifact(
        path=path,
        fingerprint=_hash(path),
        inputs=inputs,
        outputs=outputs,
        optimization_profiles=(profile,) * profiles,
        use_cuda_graphs=use_cuda_graphs,
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
    ("provider", "expected_accelerators"),
    [
        (ONNXExecutionProvider.CUDA, ()),
        (ONNXExecutionProvider.TENSORRT, ("tensorrt",)),
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
    artifact = ONNXArtifact(
        path=model_file,
        fingerprint=_hash(model_file),
        companions=(ArtifactFile(relative_path=Path("weights.bin"), fingerprint=_hash(weights)),),
        inputs=inputs,
        outputs=outputs,
        execution_provider=provider,
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
    artifact = PT2Artifact(
        path=package,
        fingerprint=_hash(package),
        inputs=(_spec("INPUT__0", DType.FLOAT32, (1, 16), (8, 16)),),
        outputs=(_spec("OUTPUT__0", DType.FLOAT32, (1, 8), (8, 8)),),
        structured_call=False,
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
    artifact = PT2Artifact(
        path=package,
        fingerprint=_hash(package),
        inputs=inputs,
        outputs=outputs,
        structured_call=True,
    )

    published = aitriton.publish(artifact, path=tmp_path / "unbatched", model_name="encoder")
    assert (published / "1" / "model.pt2").is_file()

    with pytest.raises(aitriton.PublicationError, match="structured calls"):
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

    with pytest.raises(aitriton.PublicationError, match="never replaces"):
        aitriton.publish(artifact, path=repository, model_name="encoder")

    assert (published / "config.pbtxt").read_text() == original_config


def test_integrity_failure_leaves_no_partial_model(tmp_path):
    artifact = _plan(tmp_path / "source.plan")
    artifact.path.write_bytes(b"changed")

    with pytest.raises(aitriton.PublicationError, match="changed since it was built"):
        aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")

    assert not (tmp_path / "repository" / "encoder").exists()
