# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import shlex
import sys
from pathlib import Path

from pytest import CaptureFixture
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


def test_run_script_selects_entry_and_installs_dependencies(mocker: MockerFixture, tmp_path: Path) -> None:
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
    calls: list[str] = []
    run = mocker.patch.object(
        execute.subprocess, "run", side_effect=lambda *_args, **_kwargs: calls.append("pip/script")
    )
    install_dist = mocker.patch.object(execute, "_install_dist", side_effect=lambda *_args: calls.append("dist"))

    execute.run(script, "script", 1, is_custom_docker_image=True)

    install_dist.assert_called_once_with(False, False)
    assert calls[0] == "dist"
    assert run.call_args_list[0].args[0] == [sys.executable, "-m", "pip", "install", "demo"]
    assert run.call_args_list[1].args[0] == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--pre",
        "extra",
    ]
    assert run.call_args_list[2].args[0] == [sys.executable, str(script), '--name="second"']
    assert run.call_args_list[2].kwargs["env"]["AITUNE_CONSOLE_OUTPUT"] == "1"


def test_install_dist_installs_wheels_with_dependencies(mocker: MockerFixture) -> None:
    mocker.patch.object(execute.Path, "glob", return_value=[Path("dist/aitune.whl")])
    run = mocker.patch.object(execute.subprocess, "run")

    execute._install_dist()

    run.assert_called_once_with(
        [sys.executable, "-m", "pip", "install", "dist/aitune.whl"],
        check=True,
    )


def test_run_project_uses_variant_launcher(mocker: MockerFixture, tmp_path: Path) -> None:
    project = tmp_path / "Demo"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """
[project]
name = "demo"
version = "0.1.0"

[project.scripts]
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

    assert run.call_args_list[0].args[0] == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--editable",
        str(project),
    ]
    assert run.call_args_list[1].args[0] == [
        sys.executable,
        "-m",
        "torchrun",
        "--standalone",
        "--nproc-per-node=4",
        "--module",
        "demo.inference",
        "--multi-gpu=true",
    ]
    assert run.call_args_list[1].kwargs["cwd"] == project
    assert run.call_args_list[1].kwargs["env"]["OUTPUT"] == "artifact"


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
    command = [sys.executable, str(script), '--name="demo"']
    assert capsys.readouterr().out == f"+ {shlex.join(command)}\n"
