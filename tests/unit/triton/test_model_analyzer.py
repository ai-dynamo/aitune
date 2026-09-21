# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
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
    TensorSample,
)


def _spec(name: str, dtype: DType, maximum_batch_size: int) -> BoundedTensorSpec:
    return BoundedTensorSpec(
        name=name,
        dtype=dtype,
        min_shape=(1, 8),
        max_shape=(maximum_batch_size, 8),
    )


def _with_profiles(artifact: DeploymentArtifact, *profiles: dict) -> DeploymentArtifact:
    return replace(
        artifact,
        model=replace(
            artifact.model,
            metadata={"optimization_profile_count": len(profiles), "optimization_profiles": profiles},
        ),
    )


def _plan(path: Path, *, maximum_batch_size: int = 8, profiles: int = 1) -> DeploymentArtifact:
    """Return a TensorRT artifact with a bounded implicit batch dimension."""
    path.write_bytes(b"TensorRT plan")
    artifact = DeploymentArtifact(
        model=ModelFiles(format="tensorrt_plan", path=path),
        inputs=(_spec("input", DType.FLOAT32, maximum_batch_size),),
        outputs=(_spec("output", DType.FLOAT32, maximum_batch_size),),
        runtime=RuntimeConfig(name="tensorrt"),
    )
    profile = {"input": {"min_shape": (1, 8), "opt_shape": (4, 8), "max_shape": (maximum_batch_size, 8)}}
    return _with_profiles(artifact, *((profile,) * profiles))


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
    artifact = DeploymentArtifact(
        model=ModelFiles(format="onnx", path=source),
        runtime=RuntimeConfig(name="onnxruntime", options={"execution_provider": "cuda"}),
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
    onnx = DeploymentArtifact(
        model=ModelFiles(format="onnx", path=onnx_path),
        runtime=RuntimeConfig(name="onnxruntime", options={"execution_provider": "cuda"}),
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


def test_publish_generates_analyzer_configs_automatically(tmp_path):
    artifact = _plan(tmp_path / "source.plan")
    model = aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder", max_batch_size=6)
    for name in ("fast", "manual"):
        text = (model / "model_analyzer" / f"{name}.yaml").read_text()
        config = yaml.safe_load(text)
        assert config["model_repository"] == str(model.parent.resolve())
        assert config["perf_analyzer_flags"] == {"shape": ["input:8"]}
        assert ".aitune-" not in text
        output = Path(config["output_model_repository_path"])
        assert output == tmp_path / "repository-model-analyzer" / "encoder" / f"{name}-model-repository"
        assert not output.is_relative_to(model.parent)
    assert (
        yaml.safe_load((model / "model_analyzer/fast.yaml").read_text())["run_config_search_max_model_batch_size"] == 6
    )


def test_analyzer_results_stay_outside_a_current_directory_repository(tmp_path, monkeypatch):
    artifact = _plan(tmp_path / "source.plan")
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.chdir(repository)
    model = aitriton.publish(artifact, path=".", model_name="encoder")
    for name in ("fast", "manual"):
        config = yaml.safe_load((model / "model_analyzer" / f"{name}.yaml").read_text())
        expected = tmp_path / "repository-model-analyzer" / "encoder" / f"{name}-model-repository"
        assert Path(config["output_model_repository_path"]) == expected


@pytest.mark.parametrize("batched", [False, True])
def test_publish_uses_concrete_shapes_for_dynamic_onnx_inputs(tmp_path, batched):
    source = tmp_path / "source.onnx"
    source.write_bytes(b"ONNX graph")
    artifact = DeploymentArtifact(
        model=ModelFiles(format="onnx", path=source),
        runtime=RuntimeConfig(name="onnxruntime", options={"execution_provider": "cuda"}),
        inputs=tuple(
            BoundedTensorSpec(name=name, dtype=DType.INT64, min_shape=(1, 16), max_shape=(8, 512))
            for name in ("tokens", "mask")
        ),
        outputs=(_spec("output", DType.FLOAT32, 8),),
    )
    model = aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder", dynamic_batching=batched)
    dimensions = "16" if batched else "1,16"
    for name in ("fast", "manual"):
        config = yaml.safe_load((model / "model_analyzer" / f"{name}.yaml").read_text())
        assert config["perf_analyzer_flags"] == {"shape": [f"tokens:{dimensions}", f"mask:{dimensions}"]}


def test_publish_uses_representative_backend_inputs(tmp_path):
    source = tmp_path / "source.onnx"
    source.write_bytes(b"ONNX graph")
    artifact = DeploymentArtifact(
        model=ModelFiles(format="onnx", path=source),
        runtime=RuntimeConfig(name="onnxruntime", options={"execution_provider": "cuda"}),
        inputs=(BoundedTensorSpec(name="input_ids", dtype=DType.INT64, min_shape=(1, 4), max_shape=(8, 4)),),
        outputs=(_spec("output", DType.FLOAT32, 8),),
        sample_inputs=(TensorSample(name="input_ids", shape=(2, 4), values=(17, 23, 42, 9, 3, 5, 7, 11)),),
    )

    model = aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder", max_batch_size=8)

    input_data_path = model / "model_analyzer/input-data.json"
    assert json.loads(input_data_path.read_text()) == {
        "data": [{"input_ids": {"content": [17, 23, 42, 9], "shape": [4]}}]
    }
    for name in ("fast", "manual"):
        config = yaml.safe_load((model / "model_analyzer" / f"{name}.yaml").read_text())
        assert config["perf_analyzer_flags"]["input-data"] == [str(input_data_path.resolve())]


def test_publish_uses_tensorrt_optimum_shape_and_profile_batch_bounds(tmp_path):
    artifact = replace(
        _plan(tmp_path / "source.plan"),
        inputs=(BoundedTensorSpec(name="input", dtype=DType.FLOAT32, min_shape=(1, 8), max_shape=(8, 512)),),
    )
    artifact = _with_profiles(
        artifact,
        {"input": {"min_shape": (1, 8), "opt_shape": (2, 128), "max_shape": (4, 256)}},
    )
    model = aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder", max_batch_size=8)
    fast = yaml.safe_load((model / "model_analyzer/fast.yaml").read_text())
    manual = yaml.safe_load((model / "model_analyzer/manual.yaml").read_text())
    assert fast["perf_analyzer_flags"] == {"shape": ["input:128"]}
    assert fast["run_config_search_max_model_batch_size"] == 4
    assert manual["profile_models"]["encoder"]["parameters"]["batch_sizes"] == [1, 2, 4]


def test_analyzer_generation_failure_leaves_no_published_model(tmp_path, monkeypatch):
    from aitune.triton import model_repository

    def fail(*args, **kwargs):
        raise OSError("cannot write analyzer config")

    monkeypatch.setattr(model_repository, "_write_model_analyzer_configs", fail)
    artifact = _plan(tmp_path / "source.plan")
    repository = tmp_path / "repository"
    with pytest.raises(AITunePublicationError, match="cannot write analyzer config"):
        aitriton.publish(artifact, path=repository, model_name="encoder")
    assert list(repository.iterdir()) == []
    assert not (tmp_path / "repository-model-analyzer").exists()


def test_profile_with_larger_minimum_batch_uses_explicit_search(tmp_path):
    artifact = _plan(tmp_path / "source.plan")
    artifact = _with_profiles(
        artifact,
        {"input": {"min_shape": (3, 8), "opt_shape": (4, 8), "max_shape": (8, 8)}},
    )
    model = aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder", max_batch_size=8)
    for name, batches in (("fast", [3, 8]), ("manual", [3, 4, 8])):
        config = yaml.safe_load((model / "model_analyzer" / f"{name}.yaml").read_text())
        assert config["run_config_search_mode"] == "brute"
        assert config["profile_models"]["encoder"]["parameters"]["batch_sizes"] == batches
        assert config["profile_models"]["encoder"]["model_config_parameters"]["max_batch_size"] == batches


def test_publish_pt2_generates_analyzer_configs(tmp_path):
    source = tmp_path / "model.pt2"
    source.write_bytes(b"pt2")
    artifact = DeploymentArtifact(
        model=ModelFiles(format="pt2", path=source, metadata={"structured_call": True}),
        runtime=RuntimeConfig(name="aotinductor"),
        inputs=(_spec("input", DType.FLOAT32, 1),),
        outputs=(_spec("output", DType.FLOAT32, 1),),
    )
    model = aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")
    for name in ("fast", "manual"):
        config = yaml.safe_load((model / "model_analyzer" / f"{name}.yaml").read_text())
        assert config["perf_analyzer_flags"] == {"shape": ["input:1,8"]}
        assert "run_config_search_max_model_batch_size" not in config


def test_analyzer_uses_only_enabled_tensorrt_profiles(tmp_path):
    artifact = _plan(tmp_path / "source.plan")
    artifact = replace(
        artifact,
        inputs=(BoundedTensorSpec(name="input", dtype=DType.FLOAT32, min_shape=(1, 8), max_shape=(8, 16)),),
    )
    artifact = _with_profiles(
        artifact,
        *artifact.model.metadata["optimization_profiles"],
        {"input": {"min_shape": (1, 16), "opt_shape": (4, 16), "max_shape": (8, 16)}},
    )
    model = aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder", max_batch_size=8)
    config_path = model / "config.pbtxt"
    config = text_format.Parse(config_path.read_text(), model_config_pb2.ModelConfig())
    config.instance_group[0].profile[:] = ["1"]
    config_path.write_text(text_format.MessageToString(config))

    output = aitriton.generate_model_analyzer_configs(artifact, model_path=model, path=tmp_path / "analyzer")
    for name in ("fast", "manual"):
        generated = yaml.safe_load((output / f"{name}.yaml").read_text())
        assert generated["perf_analyzer_flags"] == {"shape": ["input:16"]}
        assert generated["run_config_search_mode"] == "brute"
        groups = generated["profile_models"]["encoder"]["model_config_parameters"]["instance_group"]
        assert groups[0]["profile"] == ["1"]


@pytest.mark.parametrize("compatible_fallback", [False, True])
def test_analyzer_requires_a_common_batch_range_for_all_inputs(tmp_path, compatible_fallback):
    artifact = _plan(tmp_path / "source.plan")
    incompatible = {
        "input": {"min_shape": (1, 8), "opt_shape": (2, 8), "max_shape": (2, 8)},
        "mask": {"min_shape": (3, 8), "opt_shape": (4, 8), "max_shape": (8, 8)},
    }
    compatible = {name: {"min_shape": (4, 8), "opt_shape": (4, 8), "max_shape": (8, 8)} for name in ("input", "mask")}
    artifact = replace(
        artifact,
        inputs=(_spec("input", DType.FLOAT32, 8), _spec("mask", DType.FLOAT32, 8)),
    )
    profiles = (incompatible, compatible) if compatible_fallback else (incompatible,)
    artifact = _with_profiles(artifact, *profiles)
    repository = tmp_path / "repository"
    if not compatible_fallback:
        with pytest.raises(AITunePublicationError, match="common batch size"):
            aitriton.publish(artifact, path=repository, model_name="encoder", max_batch_size=8)
        assert list(repository.iterdir()) == []
        return

    model = aitriton.publish(artifact, path=repository, model_name="encoder", max_batch_size=8)
    for name in ("fast", "manual"):
        generated = yaml.safe_load((model / "model_analyzer" / f"{name}.yaml").read_text())
        assert generated["profile_models"]["encoder"]["parameters"]["batch_sizes"] == [4, 8]
