# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from pytest import CaptureFixture, MonkeyPatch
from pytest_mock import MockerFixture

REPO_ROOT = Path(__file__).resolve().parents[3]
EXECUTE = REPO_ROOT / "tests/functional/scripts/execute.py"


def _load_execute_module():
    spec = importlib.util.spec_from_file_location("execute", EXECUTE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


execute = _load_execute_module()


def test_run_script_selects_entry_and_installs_dependencies(
    mocker: MockerFixture, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "001_test.py"
    script.write_text(
        """
# /// script
# dependencies = ["demo"]
# arguments = [{name = "first"}, {name = "second"}]
#
# [[pip_install]]
# packages = ["extra"]
# flags = ["--pre"]
# ///
""".lstrip(),
        encoding="utf-8",
    )
    run = mocker.patch.object(execute.subprocess, "run")

    execute.run(script, "script", 1)

    assert run.call_args_list[0].args[0] == [sys.executable, "-m", "pip", "install", "--group", "functional-test"]
    assert run.call_args_list[1].args[0] == [sys.executable, "-m", "pip", "install", "demo"]
    assert run.call_args_list[2].args[0] == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--pre",
        "extra",
    ]
    freeze = run.call_args_list[3]
    assert freeze.args[0] == [sys.executable, "-m", "pip", "freeze", "--all", "--no-input", "--local", "--quiet"]
    assert Path(freeze.kwargs["stdout"].name).name == "functional_test_requirements.txt"
    assert run.call_args_list[4].args[0] == [sys.executable, str(script), "--name=second"]
    assert run.call_args_list[4].kwargs["env"]["AITUNE_CONSOLE_OUTPUT"] == "1"


def test_run_project_uses_variant_launcher(mocker: MockerFixture, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    project = tmp_path / "Demo"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """
[project]
name = "demo"
version = "0.1.0"

[project.scripts]
tune = "demo.tune:main"
inference = "demo.inference:main"

[tool.aitune]
environment = { OUTPUT = "artifact" }
variants = [
    { arguments = { multi-gpu = true }, launcher = "torchrun", processes = 4 },
]
""".strip(),
        encoding="utf-8",
    )
    run = mocker.patch.object(execute.subprocess, "run")

    execute.run(project, "project", 0)

    assert run.call_args_list[0].args[0] == [sys.executable, "-m", "pip", "install", "--group", "functional-test"]
    assert run.call_args_list[1].args[0] == [sys.executable, "-m", "pip", "install", "examples/common"]
    assert run.call_args_list[2].args[0] == [sys.executable, "-m", "pip", "install", f"{project}[dynamo]"]
    freeze = run.call_args_list[3]
    assert freeze.args[0] == [sys.executable, "-m", "pip", "freeze", "--all", "--no-input", "--local", "--quiet"]
    assert Path(freeze.kwargs["stdout"].name).name == "functional_test_requirements.txt"
    launched = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        "--module",
    ]
    assert run.call_args_list[4].args[0] == [*launched, "demo.tune", "--multi-gpu"]
    assert run.call_args_list[5].args[0] == [*launched, "demo.inference", "--multi-gpu"]
    assert run.call_args_list[4].kwargs["cwd"] == project
    assert run.call_args_list[5].kwargs["cwd"] == project
    assert run.call_args_list[4].kwargs["env"]["OUTPUT"] == "artifact"
    assert len(run.call_args_list) == 6


def test_run_project_runs_dynamo_script_when_present(
    mocker: MockerFixture, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    project = tmp_path / "Demo"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """
[project]
name = "demo"
version = "0.1.0"

[project.scripts]
tune = "demo.tune:main"
inference = "demo.inference:main"
""".strip(),
        encoding="utf-8",
    )
    (project / "run_dynamo.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    run = mocker.patch.object(execute.subprocess, "run")

    execute.run(project, "project", 0)

    assert run.call_args_list[-1].args[0] == ["./run_dynamo.sh"]
    assert run.call_args_list[-1].kwargs["cwd"] == project


def test_run_dynamo_project_workflow(mocker: MockerFixture, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    project = tmp_path / "Demo"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """
[project]
name = "demo"
version = "0.1.0"

[project.scripts]
tune = "demo.tune:main"

[tool.aitune]
workflows = ["dynamo"]
arguments = [{ image-path = "dog.webp" }]
""".strip(),
        encoding="utf-8",
    )
    (project / "run_dynamo.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    run = mocker.patch.object(execute.subprocess, "run")

    execute.run(project, "project", 0, workflow="dynamo")

    assert run.call_args_list[2].args[0] == [sys.executable, "-m", "pip", "install", f"{project}[dynamo]"]
    assert run.call_args_list[4].args[0] == [
        sys.executable,
        "-m",
        "demo.tune",
        "--image-path=dog.webp",
        "--target=python",
    ]
    assert run.call_args_list[5].args[0] == ["./run_dynamo.sh"]
    assert len(run.call_args_list) == 6


def test_run_triton_build_phase_publishes_model_repository(
    mocker: MockerFixture, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AITUNE_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    project = tmp_path / "Demo"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """
[project]
name = "demo"
version = "0.1.0"

[project.scripts]
tune = "demo.tune:main"
triton-model-store = "demo.triton.model_store:main"

[tool.aitune]
workflows = ["triton"]
triton_image = "nvcr.io/nvidia/tritonserver:26.05-py3"
variants = [{ arguments = { image-path = "dog.webp" }, launcher = "torchrun", processes = 2 }]
""".strip(),
        encoding="utf-8",
    )
    (project / "run_triton.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    run = mocker.patch.object(execute.subprocess, "run")

    execute.run(project, "project", 0, workflow="triton", phase="build")

    model_repository = tmp_path / "artifacts" / "Demo" / "model_repository"
    assert run.call_args_list[2].args[0] == [sys.executable, "-m", "pip", "install", f"{project}[triton]"]
    assert run.call_args_list[4].args[0] == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "--module",
        "demo.tune",
        "--image-path=dog.webp",
        "--target=triton",
    ]
    assert run.call_args_list[5].args[0] == [
        sys.executable,
        "-m",
        "demo.triton.model_store",
        f"--model-repository={model_repository}",
    ]
    assert len(run.call_args_list) == 6


def test_run_triton_validate_phase_runs_only_server_validation(
    mocker: MockerFixture, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AITUNE_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    project = tmp_path / "Demo"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """
[project]
name = "demo"
version = "0.1.0"

[project.scripts]
tune = "demo.tune:main"
triton-model-store = "demo.triton.model_store:main"

[tool.aitune]
workflows = ["triton"]
triton_image = "nvcr.io/nvidia/tritonserver:26.05-py3"
arguments = [{ image-path = "dog.webp" }]
""".strip(),
        encoding="utf-8",
    )
    (project / "run_triton.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    run = mocker.patch.object(execute.subprocess, "run")

    execute.run(project, "project", 0, workflow="triton", phase="validate")

    model_repository = tmp_path / "artifacts" / "Demo" / "model_repository"
    assert run.call_args_list[0].args[0] == [sys.executable, "-m", "pip", "install", "examples/common"]
    assert run.call_args_list[1].args[0] == [sys.executable, "-m", "pip", "install", f"{project}[triton]"]
    assert run.call_args_list[3].args[0] == ["./run_triton.sh", "--image-path=dog.webp"]
    assert run.call_args_list[3].kwargs["env"]["MODEL_REPOSITORY"] == str(model_repository)
    assert "TRITON_NETWORK" not in run.call_args_list[3].kwargs["env"]
    assert len(run.call_args_list) == 4


def test_run_verbose_dry_run_prints_without_executing(
    mocker: MockerFixture,
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    script = tmp_path / "001_test.py"
    script.write_text(
        """
# /// script
# arguments = [{name = "demo"}]
# ///
""".lstrip(),
        encoding="utf-8",
    )
    run = mocker.patch.object(execute.subprocess, "run")

    execute.run(script, "script", 0, verbose=True, dry_run=True)

    run.assert_not_called()
    out = capsys.readouterr().out

    assert "pip install --group functional-test" in out
    assert "pip freeze --all" in out
    assert "> functional_test_requirements.txt" in out


def test_arguments_store_true_flags_omit_false() -> None:
    assert execute._arguments({"multi-gpu": False, "quantization": True, "name": "x", "skip": None}) == [
        "--quantization",
        "--name=x",
    ]
