# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Annotation wrapper for NVTX and hardware metrics collection - one entry point for both."""

from aitune.utils.monitoring.hardware_metrics_annotation import collect_hardware_metrics
from aitune.utils.monitoring.nvtx_annotation import annotate_with_nvtx


class annotate:  # noqa: N801
    """Annotation wrapper for NVTX and hardware metrics collection - one entry point for both.

    This class wraps nvtx.annotate and collect_hardware_metrics to allow for future extensions.
    It can be used as a decorator or a context manager.

    Args:
        name (str, optional): A name associated with the annotated code range.
        color (str or int, optional): A color associated with the annotated code range.
    """

    def __init__(
        self,
        name: str | None = None,
        color: str | int | None = None,
    ):
        """Initializes the annotation."""
        self.name = name
        self.color = color

    def _post_init(self):
        """Post-initialization."""
        self._nvtx_annotation = annotate_with_nvtx(name=self.name, color=self.color)
        self._hardware_metrics_annotation = collect_hardware_metrics(self.name)

    def __enter__(self):
        """Enters the context manager."""
        if self.name is None:
            raise ValueError("Name is required for annotate as context manager")

        self._post_init()

        self._nvtx_annotation.__enter__()
        self._hardware_metrics_annotation.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exits the context manager."""
        self._hardware_metrics_annotation.__exit__(exc_type, exc_value, traceback)
        self._nvtx_annotation.__exit__(exc_type, exc_value, traceback)
        return False

    def __call__(self, func):
        """Wraps the function as a decorator."""
        if self.name is None:
            self.name = func.__name__
        self._post_init()

        return self._nvtx_annotation(self._hardware_metrics_annotation(func))
