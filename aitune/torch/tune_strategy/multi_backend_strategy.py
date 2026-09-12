# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Share backend selection because strategies can be reused across different module types."""

from abc import abstractmethod
from typing import Literal

import torch.nn as nn

from aitune.torch.backend.backend import Backend
from aitune.torch.tune_strategy.mixin.find_max_batch_size_mixin import FindMaxBatchSizeMixin
from aitune.torch.utils.module import is_distributed_module


class MultiBackendStrategy(FindMaxBatchSizeMixin):
    """Keep candidate lists in strategies and select them once the module is known.

    Rebuild defaults when switching between ordinary and distributed modules, so a
    reused strategy does not keep incompatible backends. Explicit lists always win,
    including an empty list. Each strategy must define both workflows explicitly.
    """

    def __init__(
        self,
        backends: list[Backend] | None = None,
        workflow: Literal["aot", "jit"] = "aot",
        **kwargs,
    ):
        """Defer backend construction until the module or its configuration is inspected.

        Args:
            backends: Explicit candidates, which take precedence over strategy defaults.
            workflow: Select AOT or JIT defaults before the subclass reads its configuration.
            kwargs: Arguments passed to the parent strategy.
        """
        if workflow not in ("aot", "jit"):
            raise ValueError(f"Unknown tuning workflow {workflow!r}; expected 'aot' or 'jit'.")
        super().__init__(**kwargs)
        self._backends_override = backends
        self._resolved_backends: list[Backend] | None = None
        self._distributed_defaults = False
        self._workflow = workflow
        if workflow == "jit":
            self.enable_find_max_batch_size(False)

    @classmethod
    def for_aot(cls, **kwargs) -> "MultiBackendStrategy":
        """Create a strategy using AOT candidates and its default batch-size policy.

        Match the regular constructor so existing AOT behavior stays unchanged.

        Args:
            kwargs: Strategy constructor arguments, including explicit backend overrides.
        """
        return cls(workflow="aot", **kwargs)

    @classmethod
    def for_jit(cls, **kwargs) -> "MultiBackendStrategy":
        """Create a strategy using JIT candidates and no maximum-batch-size discovery.

        Use the batches already recorded to limit tuning work during inference.

        Args:
            kwargs: Strategy constructor arguments, including explicit backend overrides.
        """
        return cls(workflow="jit", **kwargs)

    @property
    def _backends(self) -> list[Backend]:
        """Return explicit candidates or the defaults for the current module."""
        if self._backends_override is not None:
            return self._backends_override
        if self._resolved_backends is None:
            self._resolved_backends = self._resolve_default_backends()
        return self._resolved_backends

    def _configure_for_module(self, module: nn.Module) -> None:
        """Select distributed defaults without replacing user-supplied candidates."""
        if self._backends_override is not None:
            return
        distributed = is_distributed_module(module)
        if distributed != self._distributed_defaults:
            self._distributed_defaults = distributed
            self._resolved_backends = None

    def _resolve_default_backends(self) -> list[Backend]:
        """Build only the candidates for the selected workflow and module type."""
        factories = {"aot": self._default_aot_backends, "jit": self._default_jit_backends}
        return factories[self._workflow](distributed=self._distributed_defaults)

    @abstractmethod
    def _default_aot_backends(self, distributed: bool = False) -> list[Backend]:
        """Return AOT candidates that support the module's execution requirements."""
        ...

    @abstractmethod
    def _default_jit_backends(self, distributed: bool = False) -> list[Backend]:
        """Return JIT candidates that support the module's execution requirements."""
        ...
