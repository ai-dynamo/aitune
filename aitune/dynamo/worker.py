# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve AITune-tuned models as NVIDIA Dynamo endpoints."""

import asyncio
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from inspect import isawaitable
from logging import getLogger
from typing import Any

import torch.nn as nn

from aitune.dynamo.distributed import (
    is_rank_zero,
    run_collective_dynamo_worker,
    should_run_collectively,
)
from aitune.dynamo.protocol import (
    SUPPORTED_MODALITIES,
    Modality,
    get_register_model_kwargs,
    get_wiring,
    pack_response,
)
from aitune.dynamo.runtime import run_dynamo_worker

logger = getLogger(__name__)


@dataclass
class DynamoWorkerConfig:
    """Configure how an application-provided callable is exposed through Dynamo.

    The application creates and loads the callable passed to :func:`dynamo_worker`.
    This configuration only describes its Dynamo endpoint and registration.

    Args:
        type: Modality type. One of ``"image"``, ``"video"``, ``"audio"``, ``"embedding"``.
        model_path: Registration reference passed unchanged to Dynamo's
            ``register_model``. It is not used to create or load the callable.
        mapping: Optional adapter ``fn(DynamoRequest) -> dict``. The result is
            unpacked as keyword arguments when calling the model. Without a mapping,
            a plain callable receives the deserialized request object.
        namespace: Dynamo service namespace.
        component: Component name within the namespace.
        endpoint: Endpoint name within the component.
        enable_nats: Enable NATS JetStream for KV cache events.
        model_name: Optional public name advertised to clients. Defaults to the
            ``model_path`` registration reference.
    """

    type: Modality
    model_path: str
    mapping: Callable | None = None
    namespace: str = "aitune"
    component: str = "backend"
    endpoint: str = "generate"
    enable_nats: bool = False
    model_name: str | None = None


class DynamoWorker:
    """Base class for custom AITune Dynamo workers.

    Override :meth:`setup` and :meth:`serve`, then call :meth:`run`. An initialized
    multi-rank process group starts one endpoint on rank zero while every rank
    executes each request.

    Example:
        >>> import aitune.dynamo as dyn
        >>> class MyWorker(dyn.DynamoWorker):
        ...     def setup(self):
        ...         pass
        ...     async def serve(self, request):
        ...         yield request
        >>> # MyWorker().run()  # blocks until SIGTERM/SIGINT
    """

    namespace: str = "aitune"
    component: str = "backend"
    endpoint_name: str = "generate"
    _collective: bool = False

    def setup(self) -> None:
        """Initialize the model once before serving starts."""
        raise NotImplementedError

    def warmup(self) -> None:
        """Run optional model warmup after every rank completes setup."""

    async def on_ready(self, runtime: Any, endpoint: Any) -> None:
        """Run optional registration work after the endpoint is available."""

    async def serve(self, request: Any) -> AsyncGenerator[Any, None]:
        """Handle one request and yield response chunks."""
        raise NotImplementedError
        yield  # pragma: no cover

    def run(self, enable_nats: bool = False) -> None:
        """Start serving and block until shutdown.

        Args:
            enable_nats: Enable NATS JetStream for KV cache events.
        """
        on_ready = None if type(self).on_ready is DynamoWorker.on_ready else self.on_ready
        self._collective = should_run_collectively()
        runner = run_collective_dynamo_worker if self._collective else run_dynamo_worker
        runner(
            setup=self.setup,
            serve=self.serve,
            namespace=self.namespace,
            component=self.component,
            endpoint=self.endpoint_name,
            enable_nats=enable_nats,
            on_ready=on_ready,
            warmup=self.warmup,
        )


def dynamo_worker(
    model_or_fn: nn.Module | Callable,
    config: DynamoWorkerConfig,
    *,
    setup: Callable[[], None] | None = None,
    warmup: Callable[[], None] | None = None,
) -> None:
    """Serve a tuned model through a local or collective Dynamo worker.

    Args:
        model_or_fn: A callable, or a ``torch.nn.Module`` with ``config.mapping``.
        config: Modality, endpoint, request mapping, and execution configuration.
        setup: Optional rank-local model initialization. In collective execution,
            every rank must finish this callback before warmup begins.
        warmup: Optional model warmup run after setup succeeds on every rank.

    Raises:
        ValueError: If the modality is unsupported or a module has no mapping.
        ImportError: If the optional Dynamo dependencies are unavailable.

    Example:
        >>> import aitune.dynamo as dyn
        >>> import numpy as np
        >>> def embed(request):
        ...     return np.zeros((1, 768))
        >>> config = dyn.DynamoWorkerConfig(type="embedding", model_path="org/model")
        >>> # dyn.dynamo_worker(embed, config)  # blocks until shutdown
    """
    if config.type not in SUPPORTED_MODALITIES:
        raise ValueError(
            f"Unknown modality type {config.type!r}. Supported types: {', '.join(sorted(SUPPORTED_MODALITIES))}."
        )
    if isinstance(model_or_fn, nn.Module) and config.mapping is None:
        raise ValueError(
            "A mapping function is required when passing a torch.nn.Module. Provide config.mapping=fn(request) -> dict."
        )

    worker = _HighLevelDynamoWorker(model_or_fn, config, setup=setup, warmup=warmup)
    worker.run(enable_nats=config.enable_nats)


class _HighLevelDynamoWorker(DynamoWorker):
    """Adapt a callable and :class:`DynamoWorkerConfig` to the worker API."""

    def __init__(
        self,
        model_or_fn: nn.Module | Callable,
        config: DynamoWorkerConfig,
        setup: Callable[[], None] | None = None,
        warmup: Callable[[], None] | None = None,
    ) -> None:
        self._model_or_fn = model_or_fn
        self._config = config
        self._setup = setup
        self._warmup = warmup
        self.namespace = config.namespace
        self.component = config.component
        self.endpoint_name = config.endpoint
        self._wiring = None

    def setup(self) -> None:
        """Resolve protocol types on every rank before serving."""
        self._wiring = get_wiring(self._config.type)
        if self._setup is not None:
            self._setup()

    def warmup(self) -> None:
        """Run the configured warmup callback after setup succeeds on every rank."""
        if self._warmup is not None:
            self._warmup()

    async def on_ready(self, runtime: Any, endpoint: Any) -> None:
        """Register the configured model with the Dynamo frontend."""
        from dynamo.llm import register_model

        model_input, model_type, _ = self._wiring or get_wiring(self._config.type)
        served_name = self._config.model_name or self._config.model_path
        await register_model(
            model_input,
            model_type,
            endpoint,
            self._config.model_path,
            model_name=served_name,
            **get_register_model_kwargs(),
        )
        logger.info("Registered '%s' with Dynamo frontend as %s model", served_name, self._config.type)

    async def serve(self, request: Any) -> AsyncGenerator[Any, None]:
        """Deserialize one request, execute the callable, and pack rank-zero output."""
        _, _, request_cls = self._wiring or get_wiring(self._config.type)
        typed_request = request_cls(**request) if isinstance(request, dict) else request

        loop = asyncio.get_running_loop()
        if self._config.mapping is None:
            result = await loop.run_in_executor(None, lambda: self._model_or_fn(typed_request))
        else:
            kwargs = self._config.mapping(typed_request)
            if not isinstance(kwargs, dict):
                raise TypeError(f"DynamoWorkerConfig.mapping must return a dict, got {type(kwargs).__name__!r}.")
            result = await loop.run_in_executor(None, lambda: self._model_or_fn(**kwargs))

        if isawaitable(result):
            result = await result

        if self._collective and not is_rank_zero():
            return
        model_name = self._config.model_name or self._config.model_path
        yield pack_response(result, self._config.type, model_name)
