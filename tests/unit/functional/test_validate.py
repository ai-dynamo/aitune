# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATOR = REPO_ROOT / "tests/functional/scripts/validate.py"


def _load_validate_module():
    spec = importlib.util.spec_from_file_location("validate", VALIDATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validate = _load_validate_module()


def test_project_requires_tune_and_inference_scripts(tmp_path: Path) -> None:
    project = tmp_path / "pyproject.toml"
    project.write_text(
        """
[project]
name = "demo"
version = "0.1.0"

[project.scripts]
inference = "demo:main"

[tool.aitune]
""".strip(),
        encoding="utf-8",
    )

    assert validate.validate_one(project) == f"{project}: missing [project.scripts] entries: tune"


def test_project_accepts_tune_and_inference_scripts(tmp_path: Path) -> None:
    project = tmp_path / "pyproject.toml"
    project.write_text(
        """
[project]
name = "demo"
version = "0.1.0"

[project.scripts]
tune = "demo:tune"
inference = "demo:inference"

[tool.aitune]
""".strip(),
        encoding="utf-8",
    )

    assert validate.validate_one(project) is None


def test_project_accepts_dynamo_and_triton_workflow_contracts(tmp_path: Path) -> None:
    project = tmp_path / "pyproject.toml"
    project.write_text(
        """
[project]
name = "demo"
version = "0.1.0"

[project.scripts]
tune = "demo:tune"
triton-model-store = "demo:model_store"

[tool.aitune]
workflows = [{ name = "dynamo" }, { name = "triton", install_script = "install.sh" }]
""".strip(),
        encoding="utf-8",
    )
    for name in ("run_dynamo.sh", "run_triton.sh", "install.sh"):
        script = tmp_path / name
        script.write_text("#!/bin/sh\n", encoding="utf-8")
        script.chmod(0o755)

    assert validate.validate_one(project) is None


def test_project_workflow_requires_executable_script(tmp_path: Path) -> None:
    project = tmp_path / "pyproject.toml"
    project.write_text(
        """
[project]
name = "demo"
version = "0.1.0"

[project.scripts]
tune = "demo:tune"

[tool.aitune]
workflows = [{ name = "dynamo" }]
""".strip(),
        encoding="utf-8",
    )
    script = tmp_path / "run_dynamo.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    script.chmod(0o644)

    assert validate.validate_one(project) == (f"{project}: dynamo workflow requires non-executable file: run_dynamo.sh")


def test_project_workflow_requires_its_entry_points(tmp_path: Path) -> None:
    project = tmp_path / "pyproject.toml"
    project.write_text(
        """
[project]
name = "demo"
version = "0.1.0"

[project.scripts]
tune = "demo:tune"

[tool.aitune]
workflows = [{ name = "triton" }]
""".strip(),
        encoding="utf-8",
    )

    assert validate.validate_one(project) == (
        f"{project}: triton workflow requires [project.scripts] entry: triton-model-store; required file: run_triton.sh"
    )


def test_skipped_project_does_not_require_scripts(tmp_path: Path) -> None:
    project = tmp_path / "pyproject.toml"
    project.write_text(
        """
[project]
name = "demo"
version = "0.1.0"

[tool.aitune]
skip = true
""".strip(),
        encoding="utf-8",
    )

    assert validate.validate_one(project) is None
