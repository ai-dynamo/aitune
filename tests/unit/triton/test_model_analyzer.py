# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path

import pytest
import yaml

from aitune import triton as aitriton
from aitune.records import (
    BoundedTensorSpec,
    DType,
    ONNXArtifact,
    TensorRTOptimizationProfile,
    TensorRTPlanArtifact,
    TensorRTProfileInput,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spec(name: str, dtype: DType, maximum_batch_size: int) -> BoundedTensorSpec:
    return BoundedTensorSpec(
        name=name,
        dtype=dtype,
        min_shape=(1, 8),
        max_shape=(maximum_batch_size, 8),
    )


def _plan(path: Path, *, maximum_batch_size: int = 8, profiles: int = 1) -> TensorRTPlanArtifact:
    """Return a TensorRT artifact with a bounded implicit batch dimension."""
    path.write_bytes(b"TensorRT plan")
    return TensorRTPlanArtifact(
        path=path,
        fingerprint=_hash(path),
        inputs=(_spec("input", DType.FLOAT32, maximum_batch_size),),
        outputs=(_spec("output", DType.FLOAT32, maximum_batch_size),),
        optimization_profiles=(
            TensorRTOptimizationProfile(
                inputs=(
                    TensorRTProfileInput(
                        name="input",
                        min_shape=(1, 8),
                        opt_shape=(4, 8),
                        max_shape=(maximum_batch_size, 8),
                    ),
                )
            ),
        )
        * profiles,
    )


def test_generates_quick_and_manual_searches_with_tuned_bounds(tmp_path):
    artifact = _plan(tmp_path / "source.plan")
    model = aitriton.publish(
        artifact,
        path=tmp_path / "repository",
        model_name="encoder",
        max_batch_size=6,
    )
    assert "dynamic_batching" not in (model / "config.pbtxt").read_text()

    output = aitriton.generate_model_analyzer_configs(
        artifact,
        model_path=model,
        path=tmp_path / "model-analyzer",
    )

    quick = yaml.safe_load((output / "fast.yaml").read_text())
    assert quick["run_config_search_mode"] == "quick"
    assert quick["profile_models"] == ["encoder"]
    assert quick["run_config_search_min_model_batch_size"] == 1
    assert quick["run_config_search_max_model_batch_size"] == 6
    assert quick["run_config_search_min_instance_count"] == 1
    assert quick["run_config_search_max_instance_count"] == 5

    manual = yaml.safe_load((output / "manual.yaml").read_text())
    profile = manual["profile_models"]["encoder"]
    assert manual["run_config_search_mode"] == "brute"
    assert manual["run_config_search_disable"] is True
    assert profile["parameters"] == {
        "batch_sizes": [1, 2, 4, 6],
        "concurrency": [1, 2, 4, 8, 16, 32],
    }
    assert profile["model_config_parameters"] == {
        "max_batch_size": [1, 2, 4, 6],
        "instance_group": [{"kind": "KIND_GPU", "count": [1, 2, 3, 4, 5]}],
        "dynamic_batching": {"max_queue_delay_microseconds": [0, 100, 500]},
    }
    assert manual["model_repository"] == str(model.parent.resolve())
    assert manual["output_model_repository_path"] != manual["model_repository"]
    assert manual["override_output_model_repository"] is False


def test_keeps_an_unbatched_model_unbatched(tmp_path):
    source = tmp_path / "source.onnx"
    source.write_bytes(b"ONNX graph")
    artifact = ONNXArtifact(
        path=source,
        fingerprint=_hash(source),
        inputs=(_spec("input", DType.FLOAT32, 8),),
        outputs=(_spec("output", DType.FLOAT32, 8),),
    )
    model = aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")

    output = aitriton.generate_model_analyzer_configs(
        artifact,
        model_path=model,
        path=tmp_path / "model-analyzer",
    )

    quick = yaml.safe_load((output / "fast.yaml").read_text())
    assert "run_config_search_min_model_batch_size" not in quick
    assert "run_config_search_max_model_batch_size" not in quick

    manual = yaml.safe_load((output / "manual.yaml").read_text())
    profile = manual["profile_models"]["encoder"]
    assert profile["parameters"]["batch_sizes"] == [1]
    assert "max_batch_size" not in profile["model_config_parameters"]
    assert "dynamic_batching" not in profile["model_config_parameters"]


def test_refuses_an_artifact_that_does_not_match_the_published_model(tmp_path):
    plan = _plan(tmp_path / "source.plan")
    model = aitriton.publish(plan, path=tmp_path / "repository", model_name="encoder")
    onnx_path = tmp_path / "source.onnx"
    onnx_path.write_bytes(b"ONNX graph")
    onnx = ONNXArtifact(
        path=onnx_path,
        fingerprint=_hash(onnx_path),
        inputs=plan.inputs,
        outputs=plan.outputs,
    )

    with pytest.raises(aitriton.ModelAnalyzerConfigError, match="requires Triton platform"):
        aitriton.generate_model_analyzer_configs(
            onnx,
            model_path=model,
            path=tmp_path / "model-analyzer",
        )


def test_fast_search_preserves_multiple_tensorrt_profiles(tmp_path):
    artifact = _plan(tmp_path / "source.plan", profiles=2)
    model = aitriton.publish(
        artifact,
        path=tmp_path / "repository",
        model_name="encoder",
        max_batch_size=8,
    )

    output = aitriton.generate_model_analyzer_configs(
        artifact,
        model_path=model,
        path=tmp_path / "model-analyzer",
    )

    fast = yaml.safe_load((output / "fast.yaml").read_text())
    assert fast["run_config_search_mode"] == "brute"
    assert fast["profile_models"]["encoder"]["model_config_parameters"]["instance_group"] == [
        {"kind": "KIND_GPU", "count": [1, 2], "profile": ["0", "1"]}
    ]
    manual = yaml.safe_load((output / "manual.yaml").read_text())
    assert manual["profile_models"]["encoder"]["model_config_parameters"]["instance_group"] == [
        {"kind": "KIND_GPU", "count": [1, 2, 3, 4, 5], "profile": ["0", "1"]}
    ]


def test_refuses_to_replace_generated_configuration(tmp_path):
    artifact = _plan(tmp_path / "source.plan")
    model = aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")
    output = aitriton.generate_model_analyzer_configs(
        artifact,
        model_path=model,
        path=tmp_path / "model-analyzer",
    )
    original = (output / "fast.yaml").read_text()

    with pytest.raises(aitriton.ModelAnalyzerConfigError, match="never replaces"):
        aitriton.generate_model_analyzer_configs(
            artifact,
            model_path=model,
            path=output,
        )

    assert (output / "fast.yaml").read_text() == original
