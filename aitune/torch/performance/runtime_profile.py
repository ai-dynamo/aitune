# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch Profiler capture helpers for performance profiles."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import torch
from torch.autograd.profiler_util import EventList

from aitune.torch.performance.attribution_hooks import (
    _discover_aot_targets,
    _discover_untuned_targets,
    _RegionInstaller,
)
from aitune.torch.performance.context import (
    AOT_MODULE_REGION_PREFIX,
    PROFILED_RUN_REGION_NAME,
    UNTUNED_MODULE_REGION_PREFIX,
)
from aitune.torch.utils.cuda_utils import synchronize as cuda_synchronize

PROFILER_KEY_AVERAGES_ROW_LIMIT = 20


class _ProfileWarning(TypedDict):
    """Structured warning emitted in ``PerformanceProfile.data["warnings"]``."""

    code: str
    message: str
    source: str


@dataclass
class _RunTiming:
    """Per-run timing block. device_time_us is None when the profiler reported no device time."""

    wall_time_us: float
    cpu_time_us: float
    device_time_us: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON-facing dict shape, omitting unavailable device timing."""
        result: dict[str, Any] = {
            "wall_time_us": self.wall_time_us,
            "cpu_time_us": self.cpu_time_us,
        }
        if self.device_time_us is not None:
            result["device_time_us"] = self.device_time_us
        return result


@dataclass
class _RegionAggregate:
    """Sum of one region's profiler events within one profiled run. Shares are derived at to_dict() time."""

    region_id: str
    calls: int = 0
    cpu_time_us: float = 0.0
    device_time_us: float | None = None

    def accumulate(self, event: Any) -> None:
        """Fold one profiler event into this aggregate."""
        self.calls += 1
        self.cpu_time_us += _event_float(event, "cpu_time_total")
        event_device_time_us = _device_time_us(event)
        if event_device_time_us is not None:
            self.device_time_us = (self.device_time_us or 0.0) + event_device_time_us

    def to_dict(self, run_timing: _RunTiming) -> dict[str, Any]:
        """Serialize to the JSON-facing dict shape, computing shares relative to the run timing."""
        result: dict[str, Any] = {
            "region_id": self.region_id,
            "calls": self.calls,
            "cpu_time_us": self.cpu_time_us,
        }
        if run_timing.cpu_time_us > 0:
            result["cpu_time_fraction"] = self.cpu_time_us / run_timing.cpu_time_us
        if self.device_time_us is not None:
            result["device_time_us"] = self.device_time_us
            if run_timing.device_time_us is not None and run_timing.device_time_us > 0:
                result["device_time_fraction"] = self.device_time_us / run_timing.device_time_us
        return result


@dataclass
class _Residual:
    """Portion of one profiled run not attributed to any topmost-owned region, per time domain."""

    cpu_time_us: float
    cpu_time_fraction: float
    device_time_us: float | None
    device_time_fraction: float | None

    @classmethod
    def from_run(cls, timing: _RunTiming, regions: Iterable[_RegionAggregate]) -> _Residual:
        cpu_region_sum = 0.0
        device_region_sum = 0.0
        for region in regions:
            cpu_region_sum += region.cpu_time_us
            if region.device_time_us is not None:
                device_region_sum += region.device_time_us

        cpu_time_us = max(0.0, timing.cpu_time_us - cpu_region_sum)
        cpu_time_fraction = cpu_time_us / timing.cpu_time_us if timing.cpu_time_us > 0 else 0.0

        if timing.device_time_us is not None and timing.device_time_us > 0:
            device_time_us = max(0.0, timing.device_time_us - device_region_sum)
            device_time_fraction = device_time_us / timing.device_time_us
        else:
            device_time_us = None
            device_time_fraction = None

        return cls(
            cpu_time_us=cpu_time_us,
            cpu_time_fraction=cpu_time_fraction,
            device_time_us=device_time_us,
            device_time_fraction=device_time_fraction,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON-facing dict shape, omitting unavailable device timing."""
        result: dict[str, Any] = {
            "cpu_time_us": self.cpu_time_us,
            "cpu_time_fraction": self.cpu_time_fraction,
        }
        if self.device_time_us is not None:
            result["device_time_us"] = self.device_time_us
            result["device_time_fraction"] = self.device_time_fraction
        return result


@dataclass
class _RuntimeProfile:
    """Data captured from one PyTorch Profiler run plus its derived analysis."""

    run_wall_times_us: list[float]
    events: EventList
    key_averages: EventList
    activities: list[str]
    untuned_module_types: dict[str, str] = field(default_factory=dict)

    _marker_events: list[Any] = field(init=False, repr=False)
    _run_region_aggregates: dict[int, dict[str, _RegionAggregate]] = field(init=False, repr=False)
    _observed_aot_region_names: list[str] = field(init=False, repr=False)
    _observed_untuned_region_paths: list[str] = field(init=False, repr=False)
    _unmapped_aot_event_count: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._marker_events = [
            event
            for event in self.events
            if _is_cpu_profiler_event(event) and getattr(event, "name", "") == PROFILED_RUN_REGION_NAME
        ]
        if len(self._marker_events) != len(self.run_wall_times_us):
            raise RuntimeError(
                f"PyTorch Profiler produced {len(self._marker_events)} run marker event(s) "
                f"but {len(self.run_wall_times_us)} run(s) were measured."
            )
        self._classify_region_events()

    @classmethod
    def capture(
        cls,
        inference_function: Callable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        trace_file: Path | None,
        warmup_runs: int,
        measured_runs: int,
        discovery_root: Any | None = None,
    ) -> _RuntimeProfile:
        """Run warmup + measured iterations under PyTorch Profiler.

        PyTorch Profiler drives the iteration phases via ``torch.profiler.schedule``:
        ``warmup_runs`` user-requested warmup iterations during which the profiler is
        idle (no event capture, no overhead), followed by 1 profiler-internal warmup
        iteration that absorbs the profiler's buffer-allocation cost, followed by
        ``measured_runs`` recorded iterations. Only the recorded iterations contribute
        events to the captured trace and analysis.

        ``discovery_root`` selects the object from which to discover top-level untuned
        module candidates for forward-hook instrumentation. Pass ``None`` to skip
        untuned instrumentation entirely (used for pure-callable targets and tests).
        """
        activities = _profiler_activities()
        schedule = torch.profiler.schedule(skip_first=warmup_runs, wait=0, warmup=1, active=measured_runs)
        total_steps = warmup_runs + 1 + measured_runs
        first_measured_step = warmup_runs + 1
        run_wall_times_us = []

        aot_targets = _discover_aot_targets()
        untuned_targets = _discover_untuned_targets(discovery_root) if discovery_root is not None else []
        untuned_module_types = {target.path: target.module_type for target in untuned_targets}
        profiler_kwargs: dict[str, Any] = {
            "activities": activities,
            "acc_events": True,
            "schedule": schedule,
        }

        if trace_file is not None:
            trace_file.parent.mkdir(parents=True, exist_ok=True)

            def export_trace(profiler_obj: Any) -> None:
                profiler_obj.export_chrome_trace(str(trace_file))

            profiler_kwargs["on_trace_ready"] = export_trace

        with _RegionInstaller(aot_targets + untuned_targets):
            with torch.profiler.profile(**profiler_kwargs) as profiler:
                with torch.no_grad():
                    for step in range(total_steps):
                        if step < first_measured_step:
                            inference_function(*args, **kwargs)
                            cuda_synchronize()
                        else:
                            cuda_synchronize()
                            start_ns = time.perf_counter_ns()
                            with torch.profiler.record_function(PROFILED_RUN_REGION_NAME):
                                inference_function(*args, **kwargs)
                            cuda_synchronize()
                            run_wall_times_us.append((time.perf_counter_ns() - start_ns) / 1_000)
                        profiler.step()

            profiler_events = profiler.events() or EventList()
            key_averages = profiler.key_averages() or EventList()

        return cls(
            run_wall_times_us=run_wall_times_us,
            events=profiler_events,
            key_averages=key_averages,
            activities=[activity.name for activity in activities],
            untuned_module_types=untuned_module_types,
        )

    def aot_region_names(self) -> list[str]:
        """Return observed AOT-managed module region names in profiler event order."""
        return list(self._observed_aot_region_names)

    def untuned_region_paths(self) -> list[str]:
        """Return observed untuned module region paths in profiler event order."""
        return list(self._observed_untuned_region_paths)

    def runs(self) -> list[dict[str, Any]]:
        """Return per-run timing and aggregated region measurements (AOT and untuned) plus residual."""
        runs: list[dict[str, Any]] = []
        for index, wall_time_us in enumerate(self.run_wall_times_us):
            timing = self._run_timing(index, wall_time_us)
            region_aggregates = self._run_region_aggregates.get(index, {})
            runs.append({
                "run_index": index,
                "timing": timing.to_dict(),
                "regions": [aggregate.to_dict(timing) for aggregate in region_aggregates.values()],
                "residual": _Residual.from_run(timing, region_aggregates.values()).to_dict(),
            })
        return runs

    def profiler_summary(self) -> dict[str, Any]:
        """Return bounded summaries from PyTorch Profiler's own aggregate events."""
        summaries: dict[str, Any] = {
            "cpu_time_total": self._key_averages_summary(sort_by="cpu_time_total"),
        }
        if any(_event_float(event, "device_time_total") > 0 for event in self.key_averages):
            summaries["device_time_total"] = self._key_averages_summary(sort_by="device_time_total")

        return {
            "activities": self.activities,
            "key_averages": summaries,
        }

    def warnings(self) -> list[_ProfileWarning]:
        """Return structured warnings for report integrity issues."""
        warnings: list[_ProfileWarning] = []
        if self._unmapped_aot_event_count > 0:
            warnings.append(
                _ProfileWarning(
                    code="UNMAPPED_AOT_REGION_EVENTS",
                    message=(
                        f"{self._unmapped_aot_event_count} AOT region profiler event(s) "
                        "could not be mapped to a profiled run."
                    ),
                    source="core",
                )
            )
        return warnings

    def _classify_region_events(self) -> None:
        """Walk events once and bucket each region event (AOT or untuned) into its run.

        Events matching ``AOT_MODULE_REGION_PREFIX`` are tracked as AOT regions; events
        matching ``UNTUNED_MODULE_REGION_PREFIX`` are tracked as untuned regions. AOT
        events that cannot be mapped to a profiled-run ancestor are counted as
        unmapped; untuned events without a profiled-run ancestor are silently dropped
        (they fire outside the measured window, e.g. during warmup setup).

        Nested AOT events — an AOT-region event whose nearest AOT ancestor (before the
        profiled-run marker) is also an AOT-region event — are skipped from
        aggregation: the enclosing AOT region already owns that work, and counting
        both would double-count and break ``sum(fractions) + residual ≈ 1.0``. The
        wrapper's name is still recorded in ``_observed_aot_region_names`` only if at
        least one non-nested event was seen, so a wrapper that ever owned time
        appears in the regions metadata; one that only fired nested under another
        does not.
        """
        marker_indices = {id(event): index for index, event in enumerate(self._marker_events)}
        run_aggregates: dict[int, dict[str, _RegionAggregate]] = {}
        observed_aot_names: list[str] = []
        observed_aot_owned: set[str] = set()
        observed_untuned_paths: list[str] = []
        observed_untuned_seen: set[str] = set()
        unmapped_aot_count = 0

        for event in self.events:
            if not _is_cpu_profiler_event(event):
                continue
            name = getattr(event, "name", "")
            if name.startswith(AOT_MODULE_REGION_PREFIX):
                outcome = self._classify_aot_event(event, name, marker_indices, observed_aot_owned, observed_aot_names)
                if outcome is None:
                    continue
                if isinstance(outcome, str):
                    # Sentinel "unmapped" — event has no profiled-run ancestor.
                    unmapped_aot_count += 1
                    continue
                run_index, region_id = outcome
            elif name.startswith(UNTUNED_MODULE_REGION_PREFIX):
                outcome = self._classify_untuned_event(
                    event, name, marker_indices, observed_untuned_seen, observed_untuned_paths
                )
                if outcome is None:
                    continue
                run_index, region_id = outcome
            else:
                continue

            aggregates = run_aggregates.setdefault(run_index, {})
            if region_id not in aggregates:
                aggregates[region_id] = _RegionAggregate(region_id=region_id)
            aggregates[region_id].accumulate(event)

        self._observed_aot_region_names = observed_aot_names
        self._observed_untuned_region_paths = observed_untuned_paths
        self._unmapped_aot_event_count = unmapped_aot_count
        self._run_region_aggregates = run_aggregates

    @staticmethod
    def _owning_run_index(event: Any, marker_indices: dict[int, int]) -> int | None:
        """Return the profiled run index that owns an event, walking CPU parents."""
        candidate = getattr(event, "cpu_parent", None)
        while candidate is not None:
            marker_index = marker_indices.get(id(candidate))
            if marker_index is not None:
                return marker_index
            candidate = getattr(candidate, "cpu_parent", None)
        return None

    def _classify_aot_event(
        self,
        event: Any,
        name: str,
        marker_indices: dict[int, int],
        observed_owned: set[str],
        observed_names: list[str],
    ) -> tuple[int, str] | str | None:
        """Decide what to do with one AOT-prefixed event.

        Returns ``None`` if the event is nested under another AOT event (skip),
        the string ``"unmapped"`` if it has no profiled-run ancestor, or
        ``(run_index, region_id)`` for aggregation.
        """
        if self._aot_event_is_nested(event, marker_indices):
            return None
        module_name = name.removeprefix(AOT_MODULE_REGION_PREFIX)
        if module_name not in observed_owned:
            observed_owned.add(module_name)
            observed_names.append(module_name)
        run_index = self._owning_run_index(event, marker_indices)
        if run_index is None:
            return "unmapped"
        return run_index, _aot_region_id(module_name)

    def _classify_untuned_event(
        self,
        event: Any,
        name: str,
        marker_indices: dict[int, int],
        observed_seen: set[str],
        observed_paths: list[str],
    ) -> tuple[int, str] | None:
        """Decide what to do with one untuned-prefixed event."""
        module_path = name.removeprefix(UNTUNED_MODULE_REGION_PREFIX)
        if module_path not in observed_seen:
            observed_seen.add(module_path)
            observed_paths.append(module_path)
        run_index = self._owning_run_index(event, marker_indices)
        if run_index is None:
            return None
        return run_index, _untuned_region_id(module_path)

    @staticmethod
    def _aot_event_is_nested(event: Any, marker_indices: dict[int, int]) -> bool:
        """Return True if a strict CPU ancestor of ``event`` is another AOT-region event.

        Walks ``cpu_parent`` up to (but not past) the profiled-run marker. The first
        ancestor matching :data:`AOT_MODULE_REGION_PREFIX` means this event is
        already enclosed by another AOT region in this run and its time would be
        double-counted if aggregated independently.
        """
        candidate = getattr(event, "cpu_parent", None)
        while candidate is not None:
            if id(candidate) in marker_indices:
                return False
            name = getattr(candidate, "name", "")
            if name.startswith(AOT_MODULE_REGION_PREFIX):
                return True
            candidate = getattr(candidate, "cpu_parent", None)
        return False

    def _run_timing(self, index: int, wall_time_us: float) -> _RunTiming:
        """Build the timing record for one profiled run."""
        marker_event = self._marker_events[index]
        return _RunTiming(
            wall_time_us=wall_time_us,
            cpu_time_us=_event_float(marker_event, "cpu_time_total"),
            device_time_us=_device_time_us(marker_event),
        )

    def _key_averages_summary(self, sort_by: str) -> dict[str, Any]:
        """Return one bounded view over profiler key averages."""
        sorted_events = sorted(
            self.key_averages,
            key=lambda event: _event_float(event, sort_by),
            reverse=True,
        )
        return {
            "sort_by": sort_by,
            "row_limit": PROFILER_KEY_AVERAGES_ROW_LIMIT,
            "events": [
                self._serialize_key_average_row(event) for event in sorted_events[:PROFILER_KEY_AVERAGES_ROW_LIMIT]
            ],
        }

    @staticmethod
    def _serialize_key_average_row(event: Any) -> dict[str, Any]:
        """Serialize one PyTorch Profiler key-average row."""
        row = {
            "key": str(getattr(event, "key", "")),
            "device_type": _profiler_device_type(event),
            "count": int(getattr(event, "count", 0) or 0),
            "self_cpu_time_total_us": _event_float(event, "self_cpu_time_total"),
            "cpu_time_total_us": _event_float(event, "cpu_time_total"),
        }
        self_device_time_total_us = _event_float(event, "self_device_time_total")
        device_time_total_us = _event_float(event, "device_time_total")
        if self_device_time_total_us > 0:
            row["self_device_time_total_us"] = self_device_time_total_us
        if device_time_total_us > 0:
            row["device_time_total_us"] = device_time_total_us
        return row


def _profiler_activities() -> list[torch.profiler.ProfilerActivity]:
    """Return profiler activities supported by the current runtime."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def _aot_region_id(module_name: str) -> str:
    """Return the stable region id for an AOT-managed module name."""
    return f"aot_module:{module_name}"


def _untuned_region_id(module_path: str) -> str:
    """Return the stable region id for an untuned module path."""
    return f"untuned_module:{module_path}"


def _profiler_device_type(event: Any) -> str:
    """Return the profiler event device type label."""
    device_type = getattr(event, "device_type", None)
    if device_type is None:
        return "unknown"
    return str(device_type).removeprefix("DeviceType.")


def _event_float(event: Any, attr: str) -> float:
    """Read a profiler event numeric attribute as a float."""
    return float(getattr(event, attr, 0.0) or 0.0)


def _device_time_us(event: Any) -> float | None:
    """Return positive PyTorch Profiler device time in microseconds when present."""
    device_time_us = _event_float(event, "device_time_total")
    if device_time_us <= 0:
        return None
    return device_time_us


def _is_cpu_profiler_event(event: Any) -> bool:
    """Return whether a profiler event is a CPU-side event with parent ancestry."""
    device_type = getattr(event, "device_type", None)
    return device_type is None or str(device_type) == "DeviceType.CPU"
