# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardware metrics parsing and summary."""

import pandas as pd


def parse_memory_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Extract memory usage summary per module and backend (max values in GB)."""
    mem_cols = [c for c in df.columns if c.endswith("memory_used")]
    if not mem_cols:
        return pd.DataFrame(index=pd.MultiIndex.from_frame(df[["module_name", "backend"]].drop_duplicates()))
    cols = mem_cols + ["module_name", "backend"]
    result = df[cols].groupby(["module_name", "backend"]).max().div(1024**3).round(2)
    result.columns = [col.replace("_memory_used", "").title() + "\nMem [GB]" for col in result.columns]
    return result


def parse_utilization_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Extract utilization summary per module and backend (mean and max in %)."""
    util_cols = [c for c in df.columns if c.endswith("utilization")]
    if not util_cols:
        return pd.DataFrame(index=pd.MultiIndex.from_frame(df[["module_name", "backend"]].drop_duplicates()))
    cols = util_cols + ["module_name", "backend"]
    result = df[cols].groupby(["module_name", "backend"]).agg(["mean", "max"]).round(2)
    result.columns = [l1.replace("_utilization", "").title() + "\nUtil% " + l2 for l1, l2 in result.columns]
    return result


def parse_power_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Extract power usage summary per module and backend (mean and max in W)."""
    power_cols = [c for c in df.columns if c.endswith("power_usage_milliwatts")]
    if not power_cols:
        return pd.DataFrame(index=pd.MultiIndex.from_frame(df[["module_name", "backend"]].drop_duplicates()))
    cols = power_cols + ["module_name", "backend"]
    result = df[cols].groupby(["module_name", "backend"]).agg(["mean", "max"]).div(1000).round(2)
    result.columns = ["Power [W]\n" + l2 for _, l2 in result.columns]
    return result


def _split_top_level_params(params: str) -> list[str]:
    """Split a params string on commas at the top nesting level only."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in params:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current).strip())
    return parts


def _format_label_recursive(label: str, indent: str) -> str:
    """Recursively format a backend label with per-level indentation."""
    if "(" not in label:
        return label
    name, rest = label.split("(", 1)
    params = rest[:-1]  # remove only the single trailing ")"
    if not params:  # e.g. ClassName()
        return label

    child_indent = indent + "    "
    formatted_parts = []
    for part in _split_top_level_params(params):
        part = part.strip()
        if "=" in part:
            key, value = part.split("=", 1)
            formatted_parts.append(f"{key}={_format_label_recursive(value, child_indent)}")
        else:
            formatted_parts.append(_format_label_recursive(part, child_indent))

    body = f",\n{child_indent}".join(formatted_parts)
    return f"{name}(\n{child_indent}{body}\n{indent})"


def format_backend_label_for_display(label: str) -> str:
    """Format a compact backend label for multi-line table display.

    Each top-level parameter is placed on its own indented line. Nested
    config objects are formatted recursively with increased indentation.

    Example::

        TensorRTBackend(use_dynamo=False,quantization_config=ONNXQuantizationConfig(precision='int8'))
        ->
        TensorRTBackend(
            use_dynamo=False,
            quantization_config=ONNXQuantizationConfig(
                precision='int8'
            )
        )

    Labels with no parameters (e.g. ``TorchEagerBackend()``) or without
    parentheses are returned unchanged.
    """
    if "(" not in label:
        return label
    return _format_label_recursive(label, "")


def get_metrics_summary(df: pd.DataFrame) -> pd.DataFrame | None:
    """Get a formatted summary of hardware metrics (memory, utilization, power) per module and backend."""
    if df.empty or "module_name" not in df.columns:
        return None
    df = df.copy()
    df = df.dropna(subset=["module_name", "backend"])
    if df.empty:
        return None
    df_mem = parse_memory_metrics(df)
    df_util = parse_utilization_metrics(df)
    df_power = parse_power_metrics(df)
    result = pd.concat([df_mem, df_util, df_power], axis=1)
    result.index.names = ["Module", "Backend"]
    return result
