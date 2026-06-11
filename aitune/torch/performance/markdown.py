# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Markdown rendering for performance profiles."""

from __future__ import annotations

from typing import Any

ReportDict = dict[str, Any]


def _render_markdown(report: ReportDict) -> str:
    """Render a human-readable Markdown view over a performance profile."""
    sections = [
        _render_overview(report),
        _render_runs(report),
        _render_regions(report),
        _render_run_regions(report),
        _render_key_averages(report),
        _render_warnings(report),
    ]
    return "\n\n".join(section for section in sections if section) + "\n"


def _render_overview(report: ReportDict) -> str:
    config = report["config"]
    input_data = report["input"]
    target = report["target"]
    lines = [
        "# Performance Profile",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Created | `{report['created_at']}` |",
        f"| AITune Version | `{report['aitune_version']}` |",
        f"| Target | `{target['type']}` |",
        f"| Warmup Runs | {config['warmup_runs']} |",
        f"| Measured Runs | {config['measured_runs']} |",
        f"| Uses Inference Function | {_yes_no(config['uses_inference_function'])} |",
        f"| Input Args | {input_data['args_count']} |",
        f"| Input Kwargs | {_format_list(input_data['kwargs'])} |",
    ]
    return "\n".join(lines)


def _render_runs(report: ReportDict) -> str:
    rows = [
        [
            str(run["run_index"]),
            _format_us(run["timing"].get("wall_time_us")),
            _format_us(run["timing"].get("cpu_time_us")),
            _format_us(run["timing"].get("device_time_us")),
        ]
        for run in report["runs"]
    ]
    return "\n".join([
        "## Runs",
        "",
        _markdown_table(["Run", "Wall", "CPU", "Device"], rows),
    ])


def _render_regions(report: ReportDict) -> str:
    if not report["regions"]:
        return "## Regions\n\nNo regions were observed."

    rows = [
        [
            f"`{region['id']}`",
            region["name"],
            region["kind"],
            region.get("wrapper_state", "-"),
            f"`{region['module_type']}`",
        ]
        for region in report["regions"]
    ]
    return "\n".join([
        "## Regions",
        "",
        _markdown_table(["Region", "Name", "Kind", "State", "Module Type"], rows),
    ])


def _render_run_regions(report: ReportDict) -> str:
    rows = []
    for run in report["runs"]:
        run_index = str(run["run_index"])
        for region in run["regions"]:
            rows.append([
                run_index,
                f"`{region['region_id']}`",
                str(region.get("calls", 1)),
                _format_us(region.get("cpu_time_us")),
                _format_fraction(region.get("cpu_time_fraction")),
                _format_us(region.get("device_time_us")),
                _format_fraction(region.get("device_time_fraction")),
            ])
        residual = run["residual"]
        rows.append([
            run_index,
            "_(residual)_",
            "—",
            _format_us(residual.get("cpu_time_us")),
            _format_fraction(residual.get("cpu_time_fraction")),
            _format_us(residual.get("device_time_us")),
            _format_fraction(residual.get("device_time_fraction")),
        ])

    if not rows:
        return ""

    return "\n".join([
        "## Per-Run Attribution",
        "",
        _markdown_table(
            ["Run", "Region", "Calls", "CPU", "CPU Fraction", "Device", "Device Fraction"],
            rows,
        ),
    ])


def _render_key_averages(report: ReportDict) -> str:
    key_averages = report["profiler"]["key_averages"]
    sections = ["## Profiler Key Averages"]
    if "cpu_time_total" in key_averages:
        sections.append(_render_key_average_table("Sorted By CPU Total", key_averages["cpu_time_total"]))
    if "device_time_total" in key_averages:
        sections.append(_render_key_average_table("Sorted By Device Total", key_averages["device_time_total"]))
    return "\n\n".join(sections)


def _render_key_average_table(title: str, key_average: ReportDict) -> str:
    rows = [
        [
            f"`{event['key']}`",
            event["device_type"],
            str(event["count"]),
            _format_us(event.get("self_cpu_time_total_us")),
            _format_us(event.get("cpu_time_total_us")),
            _format_us(event.get("self_device_time_total_us")),
            _format_us(event.get("device_time_total_us")),
        ]
        for event in key_average["events"]
    ]
    return "\n".join([
        f"### {title}",
        "",
        _markdown_table(
            ["Name", "Device", "Count", "Self CPU", "CPU Total", "Self Device", "Device Total"],
            rows,
        ),
    ])


def _render_warnings(report: ReportDict) -> str:
    if not report["warnings"]:
        return "## Warnings\n\nNo warnings."

    rows = [
        [
            warning["code"],
            warning["source"],
            warning["message"],
        ]
        for warning in report["warnings"]
    ]
    return "\n".join([
        "## Warnings",
        "",
        _markdown_table(["Code", "Source", "Message"], rows),
    ])


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    escaped_headers = [_escape_cell(header) for header in headers]
    escaped_rows = [[_escape_cell(cell) for cell in row] for row in rows]
    lines = [
        "| " + " | ".join(escaped_headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in escaped_rows)
    return "\n".join(lines)


def _format_us(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) / 1_000:.3f} ms"


def _format_fraction(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.1f}%"


def _format_list(values: list[str]) -> str:
    if not values:
        return "-"
    return ", ".join(f"`{value}`" for value in values)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
