# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context managers that build a hierarchical tuning report."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from aitune.utils.serialization import json_serialize

if TYPE_CHECKING:
    from aitune.torch.backend.backend import Backend
    from aitune.torch.module.graph_spec import GraphSpec
    from aitune.torch.tune_strategy.tune_strategy import TuneStrategy

from aitune.torch.config import AITuneMode, config
from aitune.torch.tune_data.report_models import (
    BackendBuildReport,
    ExceptionInfo,
    GraphTuneReport,
    ModuleTuneReport,
    TuneRunReport,
)

_logger = logging.getLogger(__name__)

REPORT_FILENAME = "report.json"

_active_report: ContextVar[TuneRunReport | None] = ContextVar("aitune_report", default=None)
_active_module: ContextVar[ModuleTuneReport | None] = ContextVar("aitune_module", default=None)
_active_graph: ContextVar[GraphTuneReport | None] = ContextVar("aitune_graph", default=None)
_run_start_ts: ContextVar[float | None] = ContextVar("aitune_run_start_ts", default=None)


def _flush_active_report() -> None:
    """Write the current report snapshot to disk."""
    report = _active_report.get()
    if report is None:
        return
    try:
        path = config.tuning_data_output_dir / REPORT_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json_serialize(asdict(report))
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
    except Exception:
        _logger.debug("Failed to write tuning report", exc_info=True)


def snapshot_config(mode: AITuneMode) -> dict[str, Any]:
    """Capture relevant config attributes based on tuning mode."""
    if mode == AITuneMode.DECLARATIVE:
        return {
            "min_num_samples": config.min_num_samples,
            "max_num_samples_stored": config.max_num_samples_stored,
            "device_after_tuning": config.device_after_tuning,
            "strict_mode": config.strict_mode,
            "enable_hf_integrations": config.enable_hf_integrations,
        }
    elif mode == AITuneMode.JIT:
        from aitune.torch.jit.config import config as jit_config

        return {
            "min_samples": jit_config.min_samples,
            "batch_axis_required": jit_config.batch_axis_required,
            "max_depth_level": jit_config.max_depth_level,
            "min_parameters": jit_config.min_parameters,
            "detect_graph_breaks": jit_config.detect_graph_breaks,
            "skip_modules": jit_config.skip_modules,
            "backends": [backend.describe() for backend in jit_config.backends],
        }
    raise ValueError(f"Invalid tuning mode: {mode}")


def report_tune_run_start(mode: AITuneMode) -> None:
    """Begin a tuning run. Call :func:`report_tune_run_end` when finished."""
    if not config.enable_tuning_data_collection:
        return
    report = TuneRunReport(
        mode=mode,
        started_at=datetime.now(tz=timezone.utc),
        aitune_config=snapshot_config(mode),
    )
    _run_start_ts.set(time.perf_counter())
    _active_report.set(report)


def report_tune_run_end(exception: BaseException | None = None) -> None:
    """Finish the active tuning run and flush the report to disk."""
    report = _active_report.get()
    if report is None:
        return
    try:
        report.duration_s = time.perf_counter() - _run_start_ts.get()
        if exception is not None:
            report.exception = ExceptionInfo.from_exception(exception)
        _flush_active_report()
    except Exception:
        _logger.debug("Failed to finalise tuning report", exc_info=True)
    finally:
        _run_start_ts.set(None)
        _active_report.set(None)


@contextmanager
def report_tune_run(mode: AITuneMode):
    """Context manager that wraps a full tuning run."""
    report_tune_run_start(mode)
    try:
        yield
    except BaseException as e:
        report_tune_run_end(exception=e)
        raise
    else:
        report_tune_run_end()


@contextmanager
def report_module_tune(module_name: str, num_parameters: int):
    """Context manager that records a module-tuning span in the report."""
    report = _active_report.get()
    if report is None:
        yield
        return

    module = ModuleTuneReport(module_name=module_name, num_parameters=num_parameters)
    report.modules.append(module)
    token = _active_module.set(module)
    _flush_active_report()
    start = time.perf_counter()
    try:
        yield
    except BaseException as e:
        module.exception = ExceptionInfo.from_exception(e)
        raise
    finally:
        module.duration_s = time.perf_counter() - start
        _active_module.reset(token)
        _flush_active_report()


@contextmanager
def report_graph_tune(graph_spec: GraphSpec, strategy: TuneStrategy):
    """Context manager that records a graph-tuning span in the report."""
    module = _active_module.get()
    if module is None:
        yield {}
        return

    graph = GraphTuneReport(
        graph_name=graph_spec.name,
        input_spec=graph_spec.input_spec.to_json_dict(),
        output_spec=graph_spec.output_spec.to_json_dict(),
        strategy_name=strategy.__class__.__name__,
        strategy_config=strategy.to_json_dict(),
    )
    module.graphs.append(graph)
    token = _active_graph.set(graph)
    _flush_active_report()
    result: dict[str, Any] = {}
    start = time.perf_counter()
    try:
        yield result
    except BaseException as e:
        graph.exception = ExceptionInfo.from_exception(e)
        raise
    finally:
        graph.duration_s = time.perf_counter() - start
        graph.selected_backend = result.get("selected_backend")
        graph.strategy_results = result.get("strategy_results")
        _active_graph.reset(token)
        _flush_active_report()


@contextmanager
def report_backend_build(backend: Backend):
    """Context manager that records a backend-build span in the report."""
    graph = _active_graph.get()
    if graph is None:
        yield
        return

    backend_config = json_serialize(backend._config.to_dict()) if backend._config is not None else None
    build = BackendBuildReport(
        backend=backend.describe(),
        backend_config=backend_config,
    )
    graph.backend_builds.append(build)
    _flush_active_report()
    start = time.perf_counter()
    try:
        yield
    except BaseException as e:
        build.success = False
        build.exception = ExceptionInfo.from_exception(e)
        raise
    else:
        build.success = True
    finally:
        build.duration_s = time.perf_counter() - start
        build.build_results = backend._build_results or None
        _flush_active_report()
