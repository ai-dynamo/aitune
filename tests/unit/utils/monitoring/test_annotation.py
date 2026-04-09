# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from aitune.utils.monitoring.annotation import annotate


def test_context_manager():
    with (
        patch("aitune.utils.monitoring.annotation.annotate_with_nvtx") as mock_annotate_with_nvtx,
        patch("aitune.utils.monitoring.annotation.collect_hardware_metrics") as mock_collect_metrics,
    ):
        mock_nvtx_ctx = mock_annotate_with_nvtx.return_value
        mock_metrics_ctx = mock_collect_metrics.return_value

        with annotate(name="test"):
            pass

        # Check __enter__ calls
        mock_nvtx_ctx.__enter__.assert_called_once()
        mock_metrics_ctx.__enter__.assert_called_once()

        # Check __exit__ calls
        mock_nvtx_ctx.__exit__.assert_called_once()
        mock_metrics_ctx.__exit__.assert_called_once()


def test_context_manager_without_a_name():
    with pytest.raises(ValueError):
        with annotate():
            pass


def test_decorator():
    """Test annotate usage as a decorator."""
    with (
        patch("aitune.utils.monitoring.annotation.annotate_with_nvtx") as mock_annotate_with_nvtx,
        patch("aitune.utils.monitoring.annotation.collect_hardware_metrics") as mock_collect_metrics,
    ):
        mock_nvtx_instance = mock_annotate_with_nvtx.return_value
        mock_metrics_instance = mock_collect_metrics.return_value

        @annotate(name="test")
        def my_func():
            pass

        # check mocks called during decorating function
        mock_nvtx_instance.assert_called_once()
        mock_metrics_instance.assert_called_once()


def test_decorator_without_a_name():
    with (
        patch("aitune.utils.monitoring.annotation.annotate_with_nvtx"),
        patch("aitune.utils.monitoring.annotation.collect_hardware_metrics"),
    ):

        @annotate()
        def my_func():
            pass

        my_func()
