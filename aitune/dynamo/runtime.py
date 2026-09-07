# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local NVIDIA Dynamo endpoint lifecycle."""

import asyncio
import signal
from collections.abc import AsyncGenerator, Callable, Coroutine
from logging import getLogger
from typing import Any

logger = getLogger(__name__)


def run_dynamo_worker(
    setup: Callable[[], None],
    serve: Callable[[Any], AsyncGenerator[Any, None]],
    namespace: str = "aitune",
    component: str = "backend",
    endpoint: str = "generate",
    enable_nats: bool = False,
    on_ready: Callable[[Any, Any], Coroutine] | None = None,
    warmup: Callable[[], None] | None = None,
) -> None:
    """Run one local Dynamo endpoint until it receives SIGTERM or SIGINT."""
    worker_decorator, uvloop = _import_dynamo()

    logger.info("Setting up model...")
    setup()
    if warmup is not None:
        logger.info("Warming up model...")
        warmup()
    logger.info("Starting Dynamo endpoint %s.%s.%s", namespace, component, endpoint)

    @worker_decorator(enable_nats=enable_nats)
    async def worker(runtime: Any) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, runtime.shutdown)

        registered_endpoint = runtime.endpoint(f"{namespace}.{component}.{endpoint}")
        if on_ready is not None:
            try:
                await on_ready(runtime, registered_endpoint)
            except Exception:
                logger.exception("on_ready callback failed; shutting down")
                runtime.shutdown()
                raise

        await registered_endpoint.serve_endpoint(serve)

    uvloop.install()
    asyncio.run(worker())


def _import_dynamo() -> tuple:
    """Import optional Dynamo runtime dependencies with an actionable error."""
    try:
        import uvloop
        from dynamo.runtime import dynamo_worker

        return dynamo_worker, uvloop
    except ImportError as error:
        raise ImportError("ai-dynamo-runtime is required. Install with: uv pip install 'aitune[dynamo]'") from error
