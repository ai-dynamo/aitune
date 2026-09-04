# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATOR = REPO_ROOT / "tests/functional/scripts/generate.py"


def _load_generate_module():
    spec = importlib.util.spec_from_file_location("generate", GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generate = _load_generate_module()
FunctionalTestConfig = generate.FunctionalTestConfig
Scope = generate.Scope


def _config(metadata: dict) -> FunctionalTestConfig:
    return FunctionalTestConfig.from_toml(Path("test.py"), metadata)


def test_scope_filtering_marks_out_of_scope_jobs_as_allow_failure() -> None:
    jobs = generate._make_script_entries(
        "dataloader",
        REPO_ROOT / "tests/functional/dataloader/001_huggingface_gpt2_tokenizer_test.py",
        _config({"scope": "L2"}),
        Scope.NIGHTLY,
    )

    assert len(jobs) == 0


def test_script_arguments_expand_to_multiple_jobs() -> None:
    jobs = generate._make_script_entries(
        "pytorch",
        REPO_ROOT / "tests/functional/pytorch/001_aitune_torch_version_test.py",
        _config({"scope": "always", "arguments": [{"name": "test1"}, {"name": "test2"}]}),
        Scope.ALWAYS,
    )

    assert [job["test_number"] for job in jobs] == [0, 1]
    assert jobs[0]["id"].endswith("_001")
    assert jobs[1]["id"].endswith("_002")


def test_environment_defaults_include_console_output() -> None:
    jobs = generate._make_script_entries(
        "pytorch",
        REPO_ROOT / "tests/functional/pytorch/039_aitune_torch_profile_cuda_test.py",
        _config({}),
        Scope.ALWAYS,
    )

    assert json.loads(jobs[0]["environment"])["AITUNE_CONSOLE_OUTPUT"] == "1"


def test_unknown_tags_use_default_runner() -> None:
    jobs = generate._make_script_entries(
        "pytorch",
        REPO_ROOT / "tests/functional/pytorch/039_aitune_torch_profile_cuda_test.py",
        _config({"additional_tags": ["mem/80g", "gpu/a100"]}),
        Scope.ALWAYS,
    )

    assert jobs[0]["runner"] == "prod-aitune-tester-rtx-pro-4500-v1"


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        (["gpu"], "prod-aitune-tester-rtx-pro-4500-v1"),
        (["sm120"], "prod-aitune-tester-rtx-pro-4500-v1"),
        (["gpu/sm/120"], "prod-aitune-tester-rtx-pro-4500-v1"),
        (["anything"], "prod-aitune-tester-rtx-pro-4500-v1"),
        (["gpu/2"], "prod-aitune-tester-rtx-pro-4500-4-v1"),
        (["gpu/4"], "prod-aitune-tester-rtx-pro-4500-4-v1"),
        (["gpu/8"], "prod-aitune-tester-rtx-pro-4500-8-v1"),
    ],
)
def test_tags_map_to_runners(tags: list[str], expected: str) -> None:
    assert generate.get_runner(tags) == expected


def test_variant_runner_tag_overrides_test_tag() -> None:
    jobs = generate._make_script_entries(
        "pytorch",
        REPO_ROOT / "tests/functional/pytorch/039_aitune_torch_profile_cuda_test.py",
        _config({
            "additional_tags": ["gpu/sm/120"],
            "variants": [{"additional_tags": ["gpu/4"]}],
        }),
        Scope.ALWAYS,
    )

    assert jobs[0]["runner"] == "prod-aitune-tester-rtx-pro-4500-4-v1"


def test_custom_docker_image_is_marked_for_wheel_install() -> None:
    jobs = generate._make_script_entries(
        "pytorch",
        REPO_ROOT / "tests/functional/pytorch/039_aitune_torch_profile_cuda_test.py",
        _config({"docker_image": "custom:image"}),
        Scope.ALWAYS,
    )

    assert jobs[0]["is_custom_docker_image"] is True


def test_generate_matrix_respects_example_scope(monkeypatch, tmp_path: Path) -> None:
    examples = tmp_path / "examples"
    project_dir = examples / "Demo"
    project_dir.mkdir(parents=True)
    (project_dir / "pyproject.toml").write_text(
        """
[tool.aitune]
scope = "L2"
arguments = [{prompt = "hello"}]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("AITUNE_TEST_SCOPE", "L0")
    monkeypatch.setenv("AITUNE_EXAMPLE_SCOPE", "L1")

    matrix = generate.generate_matrix(
        script_paths=[],
        projects_paths=[str(examples)],
        default_docker_image="ghcr.io/example/aitune:sha",
        default_scope="ALWAYS",
        example_default_scope="NIGHTLY",
        test_scope_env="AITUNE_TEST_SCOPE",
        example_scope_env="AITUNE_EXAMPLE_SCOPE",
    )

    assert len(matrix) == 0


def test_generate_matrix_from_repository_is_valid_json_and_under_limit() -> None:
    matrix = generate.generate_matrix(
        script_paths=[
            "tests/functional/pytorch",
            "tests/functional/pytorch/jit",
            "tests/functional/dataloader",
            "tests/functional/dynamo",
        ],
        projects_paths=["examples"],
        default_docker_image="ghcr.io/example/aitune:sha",
        default_scope="ALWAYS",
        example_default_scope="NIGHTLY",
        test_scope_env="AITUNE_TEST_SCOPE",
        example_scope_env="AITUNE_EXAMPLE_SCOPE",
    )
    payload = json.dumps(matrix)

    assert matrix
    assert len(matrix) <= 256
    assert len(payload) < 1_000_000
    for entry in matrix:
        assert entry["runner"] in {
            "prod-aitune-tester-rtx-pro-4500-v1",
            "prod-aitune-tester-rtx-pro-4500-4-v1",
            "prod-aitune-tester-rtx-pro-4500-8-v1",
        }
        assert isinstance(entry["is_custom_docker_image"], bool)
        json.loads(entry["environment"])


def test_write_github_output(tmp_path: Path) -> None:
    output_file = tmp_path / "output"

    generate.write_github_output(output_file, [{"id": "demo"}])

    contents = output_file.read_text(encoding="utf-8")
    assert 'matrix=[{"id":"demo"}]' in contents
    assert "matrix_count=1" in contents


def test_timeout_to_minutes_parses_gitlab_style_values() -> None:
    assert generate._timeout_to_minutes("3h 20m") == 3 * 60 + 20
    assert generate._timeout_to_minutes(None) == 40


def test_generator_script_exists() -> None:
    assert GENERATOR.is_file()
