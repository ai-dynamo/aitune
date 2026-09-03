# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parse PEP-723 script metadata and ``[tool.aitune]`` project metadata."""

# /// script
# dependencies = ["pydantic"]
# ///

from __future__ import annotations

import argparse
import logging
import re
from enum import IntEnum
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger("metadata")

DEFAULT_DOCKER_IMAGE = "nvcr.io/nvidia/pytorch:25.10-py3"

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # pytype: disable=import-error


class Scope(IntEnum):
    """Scope levels for functional tests."""

    L0 = 0
    L1 = 1
    L2 = 2
    L3 = 3
    ALWAYS = 0
    DAILY = 1
    NIGHTLY = 1
    WEEKLY = 2
    MONTHLY = 3

    def in_scope(self, other: Scope) -> bool:
        """Check if this scope level is included in the given scope."""
        return self.value <= other.value


def get_scope(metadata: dict[str, Any], default_scope: str = "L0") -> Scope:
    """Get Scope object from metadata dict."""
    scope = str(metadata.get("scope", default_scope)).upper()
    try:
        return Scope(int(scope))
    except ValueError:
        pass
    try:
        return Scope[scope]
    except KeyError:
        pass
    raise ValueError(f"Invalid scope: {scope}")


def read_script_metadata(script: str, path: Path | None = None) -> dict[str, Any]:
    """Extract script metadata from a PEP-723-style Python script."""
    regex = r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$"
    matches = [match for match in re.finditer(regex, script) if match.group("type") == "script"]
    if not matches:
        return {}
    if len(matches) > 1:
        logger.warning("Multiple metadata blocks found - using the first one for %s", path)
    content = "".join(
        line[2:] if line.startswith("# ") else line[1:]
        for line in matches[0].group("content").splitlines(keepends=True)
    )
    return tomllib.loads(content)


class FunctionalVariantConfig(BaseModel):
    """Functional variant configuration."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    arguments: dict[str, Any] = Field(default_factory=dict)
    launcher: str | None = None
    processes: int | None = None
    tags: list[str] = Field(default_factory=list, validation_alias=AliasChoices("additional_tags", "tags"))
    runner: str | None = None


class FunctionalTestConfig(BaseModel):
    """Functional test or project configuration."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    path: str
    skip: bool = False
    requires_python: str | None = Field(default=None, alias="requires-python")
    scope: Scope | None = None
    timeout: str | None = None
    allow_failure: bool = False
    runner: str | None = None
    tags: list[str] = Field(default_factory=list, validation_alias=AliasChoices("additional_tags", "tags"))
    docker_image: str | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    pip_install: list[dict[str, Any]] = Field(default_factory=list)
    arguments: list[dict[str, Any]] = Field(default_factory=list)
    variants: list[FunctionalVariantConfig] = Field(default_factory=list)
    use_gated_hf_token: bool = False

    @property
    def entries(self) -> list[FunctionalVariantConfig]:
        """Get all job entries for the test or project.

        Always returns at least one entry: the default entry with no arguments.
        """
        if not self.arguments and not self.variants:
            return [FunctionalVariantConfig()]
        return [FunctionalVariantConfig.model_validate({"arguments": args}) for args in self.arguments] + self.variants

    @field_validator("scope", mode="before")
    @classmethod
    def _parse_scope(cls, value: Any) -> Scope | None:
        if value is None or isinstance(value, Scope):
            return value
        return get_scope({"scope": value})

    @field_validator("environment", mode="before")
    @classmethod
    def _stringify_environment(cls, value: Any) -> dict[str, str]:
        if not value:
            return {}
        return {str(key): str(item) for key, item in dict(value).items()}

    @classmethod
    def from_toml(
        cls,
        path: Path,
        toml_data: dict[str, Any],
        default_scope: str = "L0",
        default_docker_image: str = DEFAULT_DOCKER_IMAGE,
    ) -> FunctionalTestConfig:
        """Build from TOML dictionary metadata."""
        data = dict(toml_data)
        data["path"] = path.as_posix()
        data.setdefault("scope", default_scope)
        data.setdefault("docker_image", default_docker_image)
        return cls.model_validate(data)

    @classmethod
    def from_script(
        cls,
        script: Path,
        default_scope: str = "L0",
        default_docker_image: str = DEFAULT_DOCKER_IMAGE,
    ) -> FunctionalTestConfig:
        """Create config from a PEP-723 script file."""
        return cls.from_toml(
            script,
            read_script_metadata(script.read_text(), script),
            default_scope,
            default_docker_image,
        )

    @classmethod
    def from_project(
        cls,
        project: Path,
        default_scope: str = "L0",
        default_docker_image: str = DEFAULT_DOCKER_IMAGE,
    ) -> FunctionalTestConfig:
        """Create config from a project's TOML metadata."""
        metadata = tomllib.loads(project.read_text()).get("tool", {}).get("aitune", {})
        return cls.from_toml(project, metadata, default_scope, default_docker_image)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="PEP-723 script or pyproject.toml")
    return parser.parse_args()


def main() -> None:
    """Print parsed functional-test config as JSON."""
    logging.basicConfig(level=logging.INFO)
    path = parse_args().path
    if path.name == "pyproject.toml":
        config = FunctionalTestConfig.from_project(path)
    else:
        config = FunctionalTestConfig.from_script(path)
    print(config.model_dump_json(indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
