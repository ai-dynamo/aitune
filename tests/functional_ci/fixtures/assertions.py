# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small assertions shared by functional CI cases."""

from aitune.torch.backend import Backend
from aitune.torch.module import Module


def selected_backend(module: Module) -> Backend:
    """Return the only backend selected for a single-graph tuned module."""
    backends = module.module.backends
    assert len(backends) == 1, f"Expected one selected backend, found {len(backends)}"
    return next(iter(backends.values()))
