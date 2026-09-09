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


def _file_config(backend="onnx", **overrides):
    fields = {
        "name": "external",
        "max_batch_size": 8,
        "inputs": ({"name": "x", "data_type": "TYPE_FP32", "dims": (16,)},),
        "outputs": ({"name": "y", "data_type": "TYPE_FP32", "dims": (4,)},),
        "dynamic_batching": True,
    }
    fields.update(overrides)
    if backend == "tensorrt":
        return aitriton.TensorRTModelConfig(**fields, optimization_profile_indices=(0, 1), cuda_graphs=True)
    if backend == "pt2":
        return aitriton.TorchAOTIModelConfig(**fields, structured_call=False)
    return aitriton.ONNXRuntimeModelConfig(**fields, execution_provider=ONNXExecutionProvider.TENSORRT)


@pytest.mark.parametrize(
    ("backend", "filename"), [("onnx", "model.onnx"), ("tensorrt", "model.plan"), ("pt2", "model.pt2")]
)
def test_publish_file_preserves_config_and_copies_to_backend_layout(tmp_path, backend, filename):
    source = tmp_path / "custom_filename.bin"
    source.write_bytes(b"prebuilt model")
    config = _file_config(backend)

    published = aitriton.publish(str(source), path=tmp_path / "repository", config=config, model_version=3)

    assert published == tmp_path / "repository" / config.name
    assert (published / "3" / filename).read_bytes() == source.read_bytes()
    assert (published / "config.pbtxt").read_text() == config.to_pbtxt()
    assert source.read_bytes() == b"prebuilt model"


def test_publish_file_copies_nested_onnx_external_data(tmp_path):
    source = tmp_path / "encoder.onnx"
    source.write_bytes(b"graph")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "weights.bin").write_bytes(b"weights")

    published = aitriton.publish(
        source, path=tmp_path / "repository", config=_file_config(), companions=["data/weights.bin"]
    )

    assert (published / "1/model.onnx/model.onnx").read_bytes() == b"graph"
    assert (published / "1/model.onnx/data/weights.bin").read_bytes() == b"weights"


@pytest.mark.parametrize(
    "companions", [["../weights"], ["/weights"], ["."], ["source.onnx"], ["model.onnx"], ["a", "a"]]
)
def test_publish_file_rejects_invalid_companion_paths(tmp_path, companions):
    from aitune.exceptions import AITuneUserInputError

    with pytest.raises(AITuneUserInputError):
        aitriton.publish(
            tmp_path / "source.onnx", path=tmp_path / "repository", config=_file_config(), companions=companions
        )
    assert not (tmp_path / "repository").exists()


@pytest.mark.parametrize("backend", ["tensorrt", "pt2"])
def test_publish_file_rejects_companions_for_single_file_formats(tmp_path, backend):
    with pytest.raises(aitriton.PublicationError, match="Only ONNX"):
        aitriton.publish(
            tmp_path / "source", path=tmp_path / "repository", config=_file_config(backend), companions=["weights"]
        )


@pytest.mark.parametrize("missing_companion", [False, True])
def test_publish_file_copy_failure_leaves_no_partial_model(tmp_path, missing_companion):
    source = tmp_path / "source.onnx"
    if missing_companion:
        source.write_bytes(b"graph")
    repository = tmp_path / "repository"
    with pytest.raises(aitriton.PublicationError, match="Failed to publish"):
        aitriton.publish(
            source, path=repository, config=_file_config(), companions=["missing.bin"] if missing_companion else []
        )
    assert list(repository.iterdir()) == []


def test_publish_file_does_not_replace_existing_model(tmp_path):
    source = tmp_path / "source.onnx"
    source.write_bytes(b"original")
    published = aitriton.publish(source, path=tmp_path / "repository", config=_file_config())
    source.write_bytes(b"replacement")
    with pytest.raises(aitriton.PublicationError, match="never replaces"):
        aitriton.publish(source, path=tmp_path / "repository", config=_file_config(), model_version=2)
    assert (published / "1/model.onnx").read_bytes() == b"original"
    assert not (published / "2").exists()


@pytest.mark.parametrize(("name", "version"), [("../escape", 1), ("external", 0), ("external", True)])
def test_publish_file_rejects_invalid_target(tmp_path, name, version):
    from aitune.exceptions import AITuneUserInputError

    with pytest.raises(AITuneUserInputError):
        aitriton.publish(
            tmp_path / "source", path=tmp_path / "repository", config=_file_config(name=name), model_version=version
        )
    assert not (tmp_path / "repository").exists()


@pytest.mark.parametrize("options", [{}, {"model_name": "external"}, {"max_batch_size": 4}, {"dynamic_batching": True}])
def test_publish_file_requires_config_and_rejects_artifact_options(tmp_path, options):
    from aitune.exceptions import AITuneUserInputError

    if options:
        options = {**options, "config": _file_config()}
    with pytest.raises(AITuneUserInputError):
        aitriton.publish(tmp_path / "source", path=tmp_path / "repository", **options)
    assert not (tmp_path / "repository").exists()


@pytest.mark.parametrize("options", [{"config": _file_config()}, {"companions": ["weights"]}])
def test_publish_artifact_rejects_file_options(tmp_path, options):
    from aitune.exceptions import AITuneUserInputError

    artifact = _plan(tmp_path / "source.plan")
    with pytest.raises(AITuneUserInputError):
        aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder", **options)
    assert not (tmp_path / "repository").exists()


@pytest.mark.parametrize(
    "kind,destination", [("label", "labels.txt"), ("warmup", "warmup/data.bin"), ("state", "initial_state/data.bin")]
)
def test_publish_config_resources(tmp_path, kind, destination):
    from aitune.exceptions import AITuneUserInputError

    source = tmp_path / "source.onnx"
    source.write_bytes(b"model")
    data = tmp_path / "data.bin"
    data.write_bytes(b"sample data")
    options = {"default_model_filename": "custom.onnx"}
    if kind == "label":
        options["outputs"] = ({"name": "y", "data_type": "TYPE_FP32", "dims": (4,), "label_filename": destination},)
    elif kind == "warmup":
        options["warmup"] = (
            aitriton.ModelWarmup(
                name="warm", inputs={"x": {"data_type": "TYPE_FP32", "dims": [16], "input_data_file": "data.bin"}}
            ),
        )
    else:
        options["dynamic_batching"] = False
        options["sequence_batching"] = aitriton.SequenceBatcher(
            state=(
                {
                    "input_name": "state_in",
                    "output_name": "state_out",
                    "data_type": "TYPE_FP32",
                    "dims": [4],
                    "initial_state": [
                        {"name": "initial", "data_type": "TYPE_FP32", "dims": [4], "data_file": "data.bin"}
                    ],
                },
            )
        )
    model_config = _file_config(**options)
    repository = tmp_path / "repository"
    with pytest.raises(AITuneUserInputError, match="Missing model resources"):
        aitriton.publish(source, path=repository, config=model_config)
    assert not repository.exists()
    published = aitriton.publish(source, path=repository, config=model_config, resources={destination: data})
    assert (published / destination).read_bytes() == b"sample data"
    assert (published / "1" / "custom.onnx").read_bytes() == b"model"
    assert (
        text_format.Parse((published / "config.pbtxt").read_text(), model_config_pb2.ModelConfig())
        == model_config.to_protobuf()
    )


@pytest.mark.parametrize(
    "destination", [".", "../escape", "/absolute", "config.pbtxt", "config.pbtxt/child", "1/model.onnx"]
)
def test_publish_rejects_invalid_resource_destination(tmp_path, destination):
    from aitune.exceptions import AITuneUserInputError

    source = tmp_path / "source.onnx"
    source.write_bytes(b"model")
    with pytest.raises(AITuneUserInputError, match="Invalid model resource destination"):
        aitriton.publish(source, path=tmp_path / "repository", config=_file_config(), resources={destination: source})
    assert not (tmp_path / "repository").exists()
