# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observable coverage for publication of already built Triton models."""

import pytest
from google.protobuf import text_format
from tritonclient.grpc import model_config_pb2

from aitune.exceptions import AITunePublicationError, AITuneUserInputError
from aitune.triton import DynamicBatcher, ONNXRuntimeModelConfig, TensorRTModelConfig, TorchAOTIModelConfig, publish


def _config(backend: str):
    common = {
        "name": "external",
        "max_batch_size": 8,
        "inputs": ({"name": "x", "data_type": "TYPE_FP32", "dims": (16,)},),
        "outputs": ({"name": "y", "data_type": "TYPE_FP32", "dims": (4,)},),
        "batcher": DynamicBatcher(),
    }
    if backend == "tensorrt":
        return TensorRTModelConfig(**common, optimization_profile_indices=(0, 1))
    if backend == "pt2":
        return TorchAOTIModelConfig(**common, structured_call=False)
    return ONNXRuntimeModelConfig(**common, execution_provider="cuda")


@pytest.mark.parametrize(
    ("backend", "filename"),
    [("onnx", "model.onnx"), ("tensorrt", "model.plan"), ("pt2", "model.pt2")],
)
def test_existing_model_uses_explicit_config_and_backend_layout(tmp_path, backend, filename):
    source = tmp_path / "custom.bin"
    source.write_bytes(b"prebuilt model")
    config = _config(backend)

    model = publish(source, path=tmp_path / "repository", config=config, model_version=3)

    assert (model / "3" / filename).read_bytes() == b"prebuilt model"
    parsed = text_format.Parse((model / "config.pbtxt").read_text(), model_config_pb2.ModelConfig())
    assert parsed == config.to_protobuf()
    assert not (model / "model_analyzer").exists()


def test_existing_onnx_model_copies_external_data(tmp_path):
    source = tmp_path / "encoder.onnx"
    source.write_bytes(b"graph")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "weights.bin").write_bytes(b"weights")

    model = publish(
        source,
        path=tmp_path / "repository",
        config=_config("onnx"),
        additional_files=["data/weights.bin"],
    )

    assert (model / "1/model.onnx/model.onnx").read_bytes() == b"graph"
    assert (model / "1/model.onnx/data/weights.bin").read_bytes() == b"weights"


def test_existing_model_requires_config_and_never_replaces_model(tmp_path):
    source = tmp_path / "encoder.onnx"
    source.write_bytes(b"graph")
    repository = tmp_path / "repository"
    with pytest.raises(AITuneUserInputError, match="config is required"):
        publish(source, path=repository)
    model = publish(source, path=repository, config=_config("onnx"))
    with pytest.raises(AITunePublicationError, match="never replaces"):
        publish(source, path=repository, config=_config("onnx"))
    assert (model / "1/model.onnx").read_bytes() == b"graph"


def test_existing_model_copy_failure_can_leave_incomplete_directory(tmp_path):
    source = tmp_path / "encoder.onnx"
    source.write_bytes(b"graph")
    with pytest.raises(AITunePublicationError, match="incomplete model directory"):
        publish(
            source,
            path=tmp_path / "repository",
            config=_config("onnx"),
            additional_files=["missing.bin"],
        )
    assert (tmp_path / "repository" / "external").is_dir()


def test_existing_model_requires_and_copies_config_resources(tmp_path):
    source = tmp_path / "encoder.onnx"
    source.write_bytes(b"graph")
    labels = tmp_path / "labels.txt"
    labels.write_text("cat\ndog\n")
    config = ONNXRuntimeModelConfig(
        name="external",
        max_batch_size=8,
        execution_provider="cuda",
        inputs=({"name": "x", "data_type": "TYPE_FP32", "dims": (16,)},),
        outputs=({"name": "y", "data_type": "TYPE_FP32", "dims": (2,), "label_filename": "labels.txt"},),
    )
    repository = tmp_path / "repository"

    with pytest.raises(AITuneUserInputError, match="Missing model resources"):
        publish(source, path=repository, config=config)
    assert not repository.exists()

    model = publish(source, path=repository, config=config, resources={"labels.txt": labels})
    assert (model / "labels.txt").read_text() == "cat\ndog\n"
