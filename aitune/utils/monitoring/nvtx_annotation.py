# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Annotation for NVTX."""

from functools import wraps

import nvtx

from aitune.global_context import BACKEND_CONTEXT_KEY, MODULE_CONTEXT_KEY, global_context

NVTX_ANNOTATION_DOMAIN = "AITune"


class annotate_with_nvtx:  # noqa: N801
    """Decorator/context manager for nvtx annotations."""

    def __init__(
        self,
        name: str | None = None,
        color: str | int | None = None,
    ):  # pylint: disable=unused-argument
        """Initializes the annotation."""
        self.name = name
        self.color = color

    def _post_init(self):
        """Post-initialization."""
        name = self.name
        if global_context.get(MODULE_CONTEXT_KEY) is not None:
            name += f" module:{global_context.get(MODULE_CONTEXT_KEY)}"
        if global_context.get(BACKEND_CONTEXT_KEY) is not None:
            name += f" backend:{global_context.get(BACKEND_CONTEXT_KEY)}"

        self._nvtx_annotation = nvtx.annotate(message=name, color=self.color, domain=NVTX_ANNOTATION_DOMAIN)

    def __enter__(self):
        """Enters the context manager."""
        if self.name is None:
            raise ValueError("Name is required for annotate as context manager")

        self._post_init()
        self._nvtx_annotation.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exits the context manager."""
        self._nvtx_annotation.__exit__(exc_type, exc_value, traceback)
        return False

    def __call__(self, func):
        """Wraps the function as a decorator."""
        if self.name is None:
            self.name = func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            self._post_init()
            with self._nvtx_annotation:
                return func(*args, **kwargs)

        return wrapper
