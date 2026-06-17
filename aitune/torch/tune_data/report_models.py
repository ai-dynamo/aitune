# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataclass models defining the structured tuning report schema."""

from __future__ import annotations

import traceback as tb
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from aitune.__version__ import __version__
from aitune.torch.config import AITuneMode


@dataclass(frozen=True)
class ExceptionInfo:
    """Serializable exception info (type/message/traceback)."""

    type: str
    message: str
    traceback: str

    @classmethod
    def from_exception(cls, exc: BaseException) -> ExceptionInfo:
        """Create a persisted exception record from a Python exception."""
        traceback = "".join(tb.format_exception(type(exc), exc, exc.__traceback__))
        return cls(
            type=type(exc).__name__,
            message=str(exc),
            traceback=traceback,
        )


SCHEMA_VERSION = 2


@dataclass(kw_only=True)
class BackendBuildReport:
    """Report for a single backend build attempt."""

    backend: str
    backend_config: dict[str, Any] | None = None
    duration_s: float | None = None
    success: bool | None = None
    build_results: list[dict[str, Any]] | None = None
    log_file: str | None = None
    exception: ExceptionInfo | None = None
    throughput: float | None = None


@dataclass(kw_only=True)
class GraphTuneReport:
    """Report for tuning a single graph within a module."""

    graph_name: str
    input_spec: dict[str, Any]
    output_spec: dict[str, Any]
    strategy_name: str
    strategy_config: dict[str, Any]
    duration_s: float | None = None
    selected_backend: str | None = None
    strategy_results: list[dict[str, Any]] | None = None
    exception: ExceptionInfo | None = None
    backend_builds: list[BackendBuildReport] = field(default_factory=list)
    baseline_throughput: float | None = None


@dataclass(kw_only=True)
class ModuleTuneReport:
    """Report for tuning a single module."""

    module_name: str
    num_parameters: int
    module_id: int | None = None
    duration_s: float | None = None
    exception: ExceptionInfo | None = None
    graphs: list[GraphTuneReport] = field(default_factory=list)


@dataclass(kw_only=True)
class ModuleInspectionReport:
    """Inspection snapshot for a JIT-intercepted module."""

    module_id: int
    module_name: str | None
    module_class: str
    state: str
    level: int
    call_count: int
    num_parameters: int
    allowed_to_tune: bool
    dtypes: list[str] = field(default_factory=list)
    child_module_ids: list[int] = field(default_factory=list)
    parent_module_id: int | None = None
    parent_module_name: str | None = None
    graphs: list[dict[str, Any]] = field(default_factory=list)


@dataclass(kw_only=True)
class TuneRunReport:
    """Top-level report for a complete tuning run."""

    schema_version: int = SCHEMA_VERSION
    aitune_version: str = __version__
    mode: AITuneMode
    started_at: datetime
    aitune_config: dict[str, Any]
    duration_s: float | None = None
    exception: ExceptionInfo | None = None
    inspection_details: list[ModuleInspectionReport] = field(default_factory=list)
    modules: list[ModuleTuneReport] = field(default_factory=list)
