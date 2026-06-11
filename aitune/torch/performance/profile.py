# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Performance profile generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aitune.__version__ import __version__ as aitune_version
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.performance.markdown import _render_markdown
from aitune.torch.performance.runtime_profile import (
    _aot_region_id,
    _ProfileWarning,
    _RuntimeProfile,
    _untuned_region_id,
)
from aitune.torch.performance.utils import _qualified_type_name

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PerformanceProfile:
    """In-memory performance profile produced by :func:`profile`."""

    data: dict[str, Any]
    trace_file: Path | None = None

    def markdown(self, options: Any | None = None) -> str:
        """Render a Markdown view over the profile data."""
        if options is not None:
            raise ValueError("markdown options are not supported yet")
        return _render_markdown(self.data)


def profile(
    obj: Any,
    input_data: Any,
    *,
    inference_function: Callable | None = None,
    warmup_runs: int = 3,
    measured_runs: int = 10,
    trace_file: str | Path | None = None,
) -> PerformanceProfile:
    """Run one representative input and return an in-memory performance profile.

    The report captures warmup iterations, measured wall-clock timings, a raw
    PyTorch Profiler Chrome trace when ``trace_file`` is provided, bounded
    PyTorch Profiler key-average views, AITune-managed AOT module regions, and
    scoped untuned-module regions discovered from ``obj``. JIT attribution
    remains follow-up work. Warmup and measured invocations run under
    ``torch.no_grad()``, matching AITune inspection and tuning.

    Args:
        obj: Object to profile. Used as the callable when ``inference_function`` is not provided.
        input_data: Single representative input. A mapping is passed as kwargs, a tuple as positional args,
            ``None`` as no arguments, and any other value as one positional argument.
        inference_function: Optional callable to run instead of ``obj`` with ``input_data``.
        warmup_runs: Number of unmeasured warmup iterations during which the profiler is idle.
            One additional warmup iteration always runs inside the profiler to absorb its
            buffer-allocation cost, so the first measured run is not contaminated.
        measured_runs: Number of measured iterations.
        trace_file: Optional file path for exporting a PyTorch Profiler Chrome trace.

    Returns:
        PerformanceProfile with structured data and optional trace file path.
    """
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be greater than or equal to 0")
    if measured_runs <= 0:
        raise ValueError("measured_runs must be greater than 0")

    resolved_trace_file = Path(trace_file).expanduser().resolve() if trace_file is not None else None

    args, kwargs = _normalize_input_data(input_data)

    runtime_profile = _RuntimeProfile.capture(
        inference_function=inference_function if inference_function is not None else obj,
        args=args,
        kwargs=kwargs,
        trace_file=resolved_trace_file,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        discovery_root=obj,
    )

    aot_modules = _aot_module_metadata()
    aot_region_names = runtime_profile.aot_region_names()
    untuned_region_paths = runtime_profile.untuned_region_paths()
    regions = _regions_metadata(
        aot_region_names, aot_modules, untuned_region_paths, runtime_profile.untuned_module_types
    )

    warnings: list[_ProfileWarning] = list(runtime_profile.warnings())
    unregistered_regions = [name for name in aot_region_names if name not in aot_modules]
    if unregistered_regions:
        warnings.append(
            _ProfileWarning(
                code="UNREGISTERED_AOT_REGION",
                message=(
                    f"{len(unregistered_regions)} AOT region(s) observed in the profile have no "
                    f"registered module: {', '.join(unregistered_regions)}."
                ),
                source="core",
            )
        )

    if resolved_trace_file is not None:
        resolved_trace_file.stat()

    data = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "aitune_version": aitune_version,
        "config": {
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "uses_inference_function": inference_function is not None,
        },
        "target": {
            "type": _qualified_type_name(obj),
        },
        "input": {
            "args_count": len(args),
            "kwargs": sorted(str(key) for key in kwargs),
        },
        "runs": runtime_profile.runs(),
        "profiler": runtime_profile.profiler_summary(),
        "regions": regions,
        "warnings": warnings,
    }

    return PerformanceProfile(data=data, trace_file=resolved_trace_file)


def _normalize_input_data(input_data: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Normalize one representative input into callable args and kwargs."""
    if input_data is None:
        return (), {}
    if isinstance(input_data, Mapping):
        return (), dict(input_data)
    if isinstance(input_data, tuple):
        return input_data, {}
    return (input_data,), {}


def _regions_metadata(
    aot_region_names: list[str],
    aot_modules: dict[str, dict[str, str]],
    untuned_region_paths: list[str],
    untuned_module_types: dict[str, str],
) -> list[dict[str, Any]]:
    """Return region metadata records for both AOT-managed and untuned regions.

    AOT entries carry ``wrapper_state``; untuned entries omit it (the field is
    AOT-specific). Both share the same shape otherwise.

    Method-wrapped region paths look like ``<module_path>.<method_name>``
    (e.g. ``vae.decode``). For ``module_type`` lookup, if the full path is not in
    the metadata dict the parent path (with the trailing ``.method`` stripped) is
    tried — that's where the underlying module's type is recorded.
    """
    regions: list[dict[str, Any]] = [
        {
            "id": _aot_region_id(module_name),
            "name": module_name,
            "kind": "aot_managed_module",
            "module_type": aot_modules.get(module_name, {}).get("module_type"),
            "wrapper_state": aot_modules.get(module_name, {}).get("state", "unknown"),
        }
        for module_name in aot_region_names
    ]
    regions.extend(
        {
            "id": _untuned_region_id(module_path),
            "name": module_path,
            "kind": "untuned_module",
            "module_type": _resolve_module_type(module_path, untuned_module_types),
        }
        for module_path in untuned_region_paths
    )
    return regions


def _resolve_module_type(path: str, types: dict[str, str]) -> str | None:
    """Look up ``module_type`` for a region path, falling back to the parent path.

    Used for method-wrapped regions like ``vae.decode`` whose full path is not in
    ``types`` (which is keyed by discovered module paths). Strips the trailing
    ``.<method_name>`` and retries once.
    """
    if path in types:
        return types[path]
    if "." in path:
        parent = path.rsplit(".", 1)[0]
        if parent in types:
            return types[parent]
    return None


def _aot_module_metadata() -> dict[str, dict[str, str]]:
    """Return live declarative/AOT AITune module metadata keyed by module name."""
    modules = {}
    for name, module in MODULE_REGISTRY.modules.items():
        wrapped_module = getattr(module, "__wrapped__", module)
        modules[name] = {
            "state": module.state.value,
            "module_type": _qualified_type_name(wrapped_module),
        }
    return modules
