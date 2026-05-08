# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context managers that build a hierarchical tuning report."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aitune.utils.disk_space import check_disk_space, raise_if_out_of_space
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

_active_report: ContextVar[TuneRunReport | None] = ContextVar("aitune_report", default=None)
_active_module: ContextVar[ModuleTuneReport | None] = ContextVar("aitune_module", default=None)
_active_graph: ContextVar[GraphTuneReport | None] = ContextVar("aitune_graph", default=None)
_run_start_ts: ContextVar[float | None] = ContextVar("aitune_run_start_ts", default=None)


def _flush_active_report(path: Path | None = None) -> Path | None:
    """Write the current report snapshot to disk, returning the path written or None."""
    report = _active_report.get()
    if report is None:
        return None
    out = path if path is not None else config.tuning_data_output_path
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = json_serialize(asdict(report))
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
    except Exception as e:
        # Reporting stays best-effort for unrelated errors, but ENOSPC must surface —
        # otherwise an out-of-disk cache run looks like "tuning just didn't produce a report".
        raise_if_out_of_space(e, path=out)
        _logger.debug("Failed to write tuning report", exc_info=True)
        return None
    else:
        _logger.debug("Tuning report snapshot written to %s", out)
        return out


def snapshot_tuning_data(path: Path | None = None) -> Path | None:
    """Flush the current in-progress tuning report to disk immediately.

    Args:
        path: Destination file. Falls back to ``config.tuning_data_output_path``
            (or ``AITUNE_TUNING_DATA_PATH`` env var) when not provided.

    Returns:
        Path the report was written to, or ``None`` if no run is active or the
        write failed.
    """
    return _flush_active_report(path)


def _describe_strategy(strategy: TuneStrategy) -> dict[str, Any]:
    """Render a TuneStrategy as a JSON-friendly summary, mirroring GraphTuneReport."""
    return {"name": strategy.__class__.__name__, "config": strategy.to_json_dict()}


def snapshot_config(mode: AITuneMode) -> dict[str, Any]:
    """Capture all config attributes for the given tuning mode."""
    if mode == AITuneMode.DECLARATIVE:
        return json_serialize(config.to_dict())
    elif mode == AITuneMode.JIT:
        from aitune.torch.jit.config import config as jit_config

        snapshot = {f.name: getattr(jit_config, f.name) for f in fields(jit_config)}
        # Resolve `strategy` to the actual strategy that will run (the default when unset),
        # so the snapshot reflects reality rather than a sentinel. The strategy's own
        # `to_json_dict()` already exposes its backends list, so no separate `backends` key.
        snapshot["strategy"] = _describe_strategy(jit_config.resolve_strategy())
        return json_serialize(snapshot)
    raise ValueError(f"Invalid tuning mode: {mode}")  # pyright: ignore[reportUnreachable]


def report_tune_run_start(mode: AITuneMode) -> None:
    """Begin a tuning run. Call :func:`report_tune_run_end` when finished."""
    # Pre-flight: warn if the cache device is low on space. Runs regardless of whether
    # tuning-data collection is enabled, since the real cache writes happen either way.
    check_disk_space(config.cache_dir)
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
    except Exception as e:
        raise_if_out_of_space(e, path=config.tuning_data_output_path)
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


def report_graph_baseline_throughput(throughput: float) -> None:
    """Record the TorchEager baseline throughput on the active graph report."""
    graph = _active_graph.get()
    if graph is not None:
        graph.baseline_throughput = throughput


def report_backend_throughput(backend_description: str, throughput: float) -> None:
    """Record profiled throughput on the matching backend build report in the active graph."""
    graph = _active_graph.get()
    if graph is None:
        return
    build = next((b for b in graph.backend_builds if b.backend == backend_description), None)
    if build is not None:
        build.throughput = throughput


@contextmanager
def report_backend_build(backend: Backend, log_file: Path | None = None):
    """Context manager that records a backend-build span in the report."""
    graph = _active_graph.get()
    if graph is None:
        yield
        return

    backend_config = json_serialize(backend._config.to_dict()) if backend._config is not None else None
    build = BackendBuildReport(
        backend=backend.describe(),
        backend_config=backend_config,
        log_file=str(log_file) if log_file is not None else None,
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
