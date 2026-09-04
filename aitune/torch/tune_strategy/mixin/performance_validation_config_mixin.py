# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared performance validation configuration for tune strategies."""

from aitune.torch.tune_strategy.performance_validation import PerformanceValidationMode
from aitune.torch.tune_strategy.tune_strategy import TuneStrategy


class PerformanceValidationConfigMixin(TuneStrategy):
    """Provides performance validation mode configuration without profiling behavior."""

    def __init__(
        self,
        *args,
        performance_validation_mode: PerformanceValidationMode | str = PerformanceValidationMode.ENFORCED,
        **kwargs,
    ):
        """Initialize the performance validation mode.

        Args:
            *args: Positional arguments forwarded through cooperative multiple inheritance.
            performance_validation_mode: Whether to disable validation, collect diagnostics only, or enforce
                the comparison with eager. Defaults to enforced.
            **kwargs: Keyword arguments forwarded through cooperative multiple inheritance.
        """
        super().__init__(*args, **kwargs)
        self._performance_validation_mode = PerformanceValidationMode(performance_validation_mode)

    def enable_performance_validation(self, enable: bool = True) -> "PerformanceValidationConfigMixin":
        """Enable enforced validation or disable eager-baseline validation.

        This compatibility method maps ``True`` to :attr:`PerformanceValidationMode.ENFORCED` and ``False``
        to :attr:`PerformanceValidationMode.DISABLED`. Use :meth:`set_performance_validation_mode` to collect
        diagnostic metrics without enforcing the eager comparison.
        """
        self._performance_validation_enabled = enable
        return self

    def set_performance_validation_mode(
        self, mode: PerformanceValidationMode | str
    ) -> "PerformanceValidationConfigMixin":
        """Set how eager-baseline performance data affects backend selection."""
        self._performance_validation_mode = PerformanceValidationMode(mode)
        return self

    @property
    def performance_validation_mode(self) -> PerformanceValidationMode:
        """Return the configured performance validation mode."""
        return self._performance_validation_mode

    @property
    def _performance_validation_enabled(self) -> bool:
        """Return whether eager-baseline validation is enabled."""
        return self._performance_validation_mode is not PerformanceValidationMode.DISABLED

    @_performance_validation_enabled.setter
    def _performance_validation_enabled(self, enable: bool) -> None:
        """Map the legacy boolean flag to disabled or enforced mode."""
        self._performance_validation_mode = (
            PerformanceValidationMode.ENFORCED if enable else PerformanceValidationMode.DISABLED
        )
