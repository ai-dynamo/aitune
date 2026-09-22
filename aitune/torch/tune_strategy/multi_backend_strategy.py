# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base class for strategies that evaluate multiple explicit backends."""

from aitune.torch.backend import Backend
from aitune.torch.tune_strategy.mixin.find_max_batch_size_mixin import FindMaxBatchSizeMixin


class MultiBackendStrategy(FindMaxBatchSizeMixin):
    """Run a selection algorithm over backends supplied by the caller."""

    def __init__(
        self,
        backends: list[Backend],
        **kwargs,
    ):
        """Store the backend candidates evaluated by this strategy.

        Args:
            backends: Explicit backend candidates.
            kwargs: Arguments passed to the parent strategy.
        """
        super().__init__(**kwargs)
        self._backends = backends
