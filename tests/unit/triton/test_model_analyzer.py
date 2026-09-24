# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from base64 import b64encode
from dataclasses import replace
from pathlib import Path
from struct import pack

import pytest
import yaml
from pydantic import ValidationError

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


def test_generates_quick_search_with_tuned_bounds_and_concurrency(tmp_path):
    artifact = _plan(tmp_path / "source.plan")
    model = aitriton.publish(
        artifact,
        path=tmp_path / "repository",
        model_name="encoder",
    )
    assert "dynamic_batching" in (model / "config.pbtxt").read_text()

    output = model / "model_analyzer"

    quick = yaml.safe_load((output / "config.yaml").read_text())
    assert quick["run_config_search_mode"] == "quick"
    assert quick["profile_models"] == ["encoder"]
    assert quick["run_config_search_min_model_batch_size"] == 1
    assert quick["run_config_search_max_model_batch_size"] == 8
    assert quick["run_config_search_min_instance_count"] == 1
    assert quick["run_config_search_max_instance_count"] == 3
    assert quick["run_config_search_max_concurrency"] == 16
    assert "constraints" not in quick
    assert "perf_latency_p95" in quick["inference_output_fields"]
    assert quick["perf_analyzer_flags"]["percentile"] == 95
    assert {path.name for path in output.iterdir()} == {"config.yaml"}


def test_publish_preserves_optional_model_analyzer_latency_budget(tmp_path):
    artifact = _plan(tmp_path / "source.plan")
    model = aitriton.publish(
        artifact,
        path=tmp_path / "repository",
        model_name="encoder",
        model_analyzer=aitriton.ModelAnalyzerConfig(latency_budget_ms=50),
    )

    config = yaml.safe_load((model / "model_analyzer/config.yaml").read_text())
    assert config["constraints"] == {"perf_latency_p95": {"max": 50}}
    assert "latency_budget" not in config


def test_publish_uses_configured_model_analyzer_percentile(tmp_path):
    artifact = _plan(tmp_path / "source.plan")
    model = aitriton.publish(
        artifact,
        path=tmp_path / "repository",
        model_name="encoder",
        model_analyzer=aitriton.ModelAnalyzerConfig(latency_budget_ms=50, latency_percentile=90),
    )

    config = yaml.safe_load((model / "model_analyzer/config.yaml").read_text())
    assert config["perf_analyzer_flags"]["percentile"] == 90
    assert config["constraints"] == {"perf_latency_p90": {"max": 50}}
    assert "perf_latency_p90" in config["inference_output_fields"]


def test_publish_stabilizes_on_configured_percentile_without_budget(tmp_path):
    artifact = _plan(tmp_path / "source.plan")
    model = aitriton.publish(
        artifact,
        path=tmp_path / "repository",
        model_name="encoder",
        model_analyzer=aitriton.ModelAnalyzerConfig(latency_percentile=99),
    )

    config = yaml.safe_load((model / "model_analyzer/config.yaml").read_text())
    assert config["perf_analyzer_flags"]["percentile"] == 99
    assert "constraints" not in config


def test_publish_rejects_unavailable_model_analyzer_percentile(tmp_path):
    with pytest.raises(ValidationError, match="latency_percentile"):
        aitriton.ModelAnalyzerConfig(latency_percentile=97)


def test_publish_rejects_invalid_model_analyzer_latency_budget(tmp_path):
    with pytest.raises(ValidationError, match="latency_budget_ms"):
        aitriton.ModelAnalyzerConfig(latency_budget_ms=0)


def test_keeps_an_unbatched_model_unbatched(tmp_path):
    source = tmp_path / "source.onnx"
    source.write_bytes(b"ONNX graph")
    artifact = DeploymentArtifact(
        model=ModelFiles(format="onnx", path=source),
        runtime=RuntimeConfig(name="onnxruntime", options={"execution_provider": "cuda"}),
        inputs=(replace(_spec("input", DType.FLOAT32, 8), batch_axis=None),),
        outputs=(replace(_spec("output", DType.FLOAT32, 8), batch_axis=None),),
    )
    model = aitriton.publish(
        artifact,
        path=tmp_path / "repository",
        model_name="encoder",
    )

    output = model / "model_analyzer"

    quick = yaml.safe_load((output / "config.yaml").read_text())
    assert "run_config_search_min_model_batch_size" not in quick
    assert "run_config_search_max_model_batch_size" not in quick
    assert quick["run_config_search_max_concurrency"] == 2


def test_analyzer_uses_quick_search_for_multiple_tensorrt_profiles(tmp_path):
    artifact = _plan(tmp_path / "source.plan", profiles=2)
    artifact = _with_profiles(
        artifact,
        {"input": {"min_shape": (3, 8), "opt_shape": (4, 8), "max_shape": (5, 8)}},
        {"input": {"min_shape": (6, 8), "opt_shape": (7, 8), "max_shape": (8, 8)}},
    )
    model = aitriton.publish(
        artifact,
        path=tmp_path / "repository",
        model_name="encoder",
    )

    output = model / "model_analyzer"

    config = yaml.safe_load((output / "config.yaml").read_text())
    assert config["run_config_search_mode"] == "quick"
    assert config["perf_analyzer_flags"] == {"shape": ["input:8"], "percentile": 95}
    assert config["profile_models"] == ["encoder"]
    assert config["run_config_search_max_model_batch_size"] == 8
    assert config["run_config_search_max_concurrency"] == 16


def test_publish_generates_analyzer_config_automatically(tmp_path):
    artifact = _plan(tmp_path / "source.plan")
    model = aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")
    text = (model / "model_analyzer/config.yaml").read_text()
    config = yaml.safe_load(text)
    assert config["model_repository"] == str(model.parent.resolve())
    assert config["perf_analyzer_flags"] == {"shape": ["input:8"], "percentile": 95}
    assert config["run_config_search_max_model_batch_size"] == 8
    assert config["run_config_search_max_concurrency"] == 16
    assert ".aitune-" not in text
    output = Path(config["output_model_repository_path"])
    assert output == tmp_path / "repository-model-analyzer" / "encoder" / "model-repository"
    assert not output.is_relative_to(model.parent)


def test_analyzer_results_stay_outside_a_current_directory_repository(tmp_path, monkeypatch):
    artifact = _plan(tmp_path / "source.plan")
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.chdir(repository)
    model = aitriton.publish(artifact, path=".", model_name="encoder")
    config = yaml.safe_load((model / "model_analyzer/config.yaml").read_text())
    expected = tmp_path / "repository-model-analyzer" / "encoder" / "model-repository"
    assert Path(config["output_model_repository_path"]) == expected


@pytest.mark.parametrize("batched", [False, True])
def test_publish_uses_concrete_shapes_for_dynamic_onnx_inputs(tmp_path, batched):
    source = tmp_path / "source.onnx"
    source.write_bytes(b"ONNX graph")
    artifact = DeploymentArtifact(
        model=ModelFiles(format="onnx", path=source),
        runtime=RuntimeConfig(name="onnxruntime", options={"execution_provider": "cuda"}),
        inputs=tuple(
            BoundedTensorSpec(
                name=name, dtype=DType.INT64, min_shape=(1, 16), max_shape=(8, 512), batch_axis=0 if batched else None
            )
            for name in ("tokens", "mask")
        ),
        outputs=(replace(_spec("output", DType.FLOAT32, 8), batch_axis=0 if batched else None),),
    )
    model = aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")
    dimensions = "16" if batched else "1,16"
    config = yaml.safe_load((model / "model_analyzer/config.yaml").read_text())
    assert config["perf_analyzer_flags"] == {
        "shape": [f"tokens:{dimensions}", f"mask:{dimensions}"],
        "percentile": 95,
    }


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

    model = aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")

    input_data_path = model / "model_analyzer/input-data.json"
    assert json.loads(input_data_path.read_text()) == {
        "data": [{"input_ids": {"content": [17, 23, 42, 9], "shape": [4]}}]
    }
    config = yaml.safe_load((model / "model_analyzer/config.yaml").read_text())
    assert config["perf_analyzer_flags"]["input-data"] == [str(input_data_path.resolve())]


def test_publish_encodes_fp16_representative_inputs_as_binary(tmp_path):
    values = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    artifact = replace(
        _plan(tmp_path / "source.plan"),
        inputs=(_spec("input", DType.FLOAT16, 8),),
        sample_inputs=(TensorSample(name="input", shape=(1, 8), values=values),),
    )

    model = aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")

    input_data_path = model / "model_analyzer/input-data.json"
    assert json.loads(input_data_path.read_text()) == {
        "data": [{"input": {"content": {"b64": b64encode(pack("<8e", *values)).decode("ascii")}, "shape": [8]}}]
    }
    config = yaml.safe_load((model / "model_analyzer/config.yaml").read_text())
    assert config["perf_analyzer_flags"]["input-data"] == [str(input_data_path.resolve())]


def test_analyzer_uses_artifact_shapes_and_batch_bounds_for_tensorrt(tmp_path):
    artifact = replace(
        _plan(tmp_path / "source.plan"),
        inputs=(BoundedTensorSpec(name="input", dtype=DType.FLOAT32, min_shape=(1, 8), max_shape=(8, 512)),),
    )
    artifact = _with_profiles(
        artifact,
        {"input": {"min_shape": (1, 8), "opt_shape": (2, 128), "max_shape": (4, 256)}},
    )
    model = aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")
    config = yaml.safe_load((model / "model_analyzer/config.yaml").read_text())
    assert config["perf_analyzer_flags"] == {"shape": ["input:8"], "percentile": 95}
    assert config["run_config_search_max_model_batch_size"] == 8
    assert config["run_config_search_max_concurrency"] == 16


def test_analyzer_generation_failure_leaves_partial_model(tmp_path, monkeypatch):
    from aitune.triton import model_repository

    def fail(*args, **kwargs):
        raise OSError("cannot write analyzer config")

    monkeypatch.setattr(model_repository, "write_model_analyzer_config", fail)
    artifact = _plan(tmp_path / "source.plan")
    repository = tmp_path / "repository"
    with pytest.raises(AITunePublicationError, match="cannot write analyzer config"):
        aitriton.publish(artifact, path=repository, model_name="encoder")
    assert (repository / "encoder" / "1" / "model.plan").is_file()
    assert not (repository / "encoder" / "model_analyzer" / "config.yaml").exists()
    assert not (tmp_path / "repository-model-analyzer").exists()


def test_analyzer_ignores_tensorrt_profile_batch_bounds(tmp_path):
    artifact = _plan(tmp_path / "source.plan")
    artifact = _with_profiles(
        artifact,
        {"input": {"min_shape": (3, 8), "opt_shape": (4, 8), "max_shape": (8, 8)}},
    )
    model = aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")
    config = yaml.safe_load((model / "model_analyzer/config.yaml").read_text())
    assert config["run_config_search_mode"] == "quick"
    assert config["run_config_search_min_model_batch_size"] == 1
    assert config["run_config_search_max_model_batch_size"] == 8


def test_publish_pt2_generates_analyzer_config(tmp_path):
    source = tmp_path / "model.pt2"
    source.write_bytes(b"pt2")
    artifact = DeploymentArtifact(
        model=ModelFiles(format="pt2", path=source, metadata={"structured_call": True}),
        runtime=RuntimeConfig(name="aotinductor"),
        inputs=(_spec("input", DType.FLOAT32, 1),),
        outputs=(_spec("output", DType.FLOAT32, 1),),
    )
    model = aitriton.publish(
        artifact,
        path=tmp_path / "repository",
        model_name="encoder",
    )
    config = yaml.safe_load((model / "model_analyzer/config.yaml").read_text())
    assert config["perf_analyzer_flags"] == {"shape": ["input:1,8"], "percentile": 95}
    assert "run_config_search_max_model_batch_size" not in config


def test_analyzer_ignores_disjoint_tensorrt_profile_input_ranges(tmp_path):
    artifact = _plan(tmp_path / "source.plan")
    incompatible = {
        "input": {"min_shape": (1, 8), "opt_shape": (2, 8), "max_shape": (2, 8)},
        "mask": {"min_shape": (3, 8), "opt_shape": (4, 8), "max_shape": (8, 8)},
    }
    artifact = replace(
        artifact,
        inputs=(_spec("input", DType.FLOAT32, 8), _spec("mask", DType.FLOAT32, 8)),
    )
    artifact = _with_profiles(artifact, incompatible)
    model = aitriton.publish(artifact, path=tmp_path / "repository", model_name="encoder")
    config = yaml.safe_load((model / "model_analyzer/config.yaml").read_text())
    assert config["run_config_search_mode"] == "quick"
    assert config["run_config_search_max_model_batch_size"] == 8
