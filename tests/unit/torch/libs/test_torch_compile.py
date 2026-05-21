# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for shared torch.compile helpers."""

from unittest.mock import Mock

from aitune.torch.libs.torch_compile import resolve_compile_dynamic


def _graph_spec_with_dynamic_axis(dynamic_axis: bool):
    graph_spec = Mock()
    graph_spec.input_spec.detected_dynamic_axis.return_value = dynamic_axis
    return graph_spec


def test_resolve_compile_dynamic_keeps_static_auto_dynamic_as_none():
    assert resolve_compile_dynamic(None, _graph_spec_with_dynamic_axis(False)) is None


def test_resolve_compile_dynamic_enables_dynamic_for_dynamic_graph_spec():
    assert resolve_compile_dynamic(None, _graph_spec_with_dynamic_axis(True)) is True


def test_resolve_compile_dynamic_preserves_explicit_dynamic_false():
    assert resolve_compile_dynamic(False, _graph_spec_with_dynamic_axis(True)) is False
