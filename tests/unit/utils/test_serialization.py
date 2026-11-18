# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for serialization helpers."""

import logging
from dataclasses import dataclass

import pytest

from aitune.utils.serialization import drop_unsupported_fields


@dataclass
class LocalConfig:
    count: int = 0
    values: list | None = None


@pytest.mark.parametrize("config", [{}, {"count": 3, "values": [1, 2]}])
def test_drop_unsupported_fields_keeps_supported_keys(config, caplog):
    with caplog.at_level(logging.WARNING):
        result = drop_unsupported_fields(LocalConfig, config)
    assert result == config
    assert all(result[key] is value for key, value in config.items())
    assert not caplog.records


def test_drop_unsupported_fields_warns_about_dropped_keys(caplog):
    config = {"count": 3, "old": True, "removed": 4}
    with caplog.at_level(logging.WARNING):
        result = drop_unsupported_fields(LocalConfig, config)
    assert result == {"count": 3}
    assert caplog.record_tuples == [
        (
            "aitune.utils.serialization",
            logging.WARNING,
            "Ignoring LocalConfig settings not supported by the installed version: ['old', 'removed']",
        )
    ]
