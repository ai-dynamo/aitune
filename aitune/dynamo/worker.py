# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamo worker — serve AITune-tuned models as Dynamo endpoints."""

import asyncio
import base64
import dataclasses
import signal
import time
from collections.abc import AsyncGenerator, Callable, Coroutine
from logging import getLogger
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import torch
import torch.nn as nn

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# DynamoWorkerConfig
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DynamoWorkerConfig:
    """Configuration for the high-level :func:`dynamo_worker` entrypoint.

    Args:
        type: Modality type. One of ``"image"``, ``"video"``, ``"embedding"``.
        model_path: HuggingFace model ID or local path passed to ``register_model``.
        mapping: Optional adapter ``fn(DynamoRequest) -> dict``. The dict is
            unpacked as ``**kwargs`` when calling the user function. If ``None``
            and the user passed a plain callable (not ``nn.Module``), the raw
            Dynamo request object is passed as the sole positional argument.
        namespace: Dynamo service namespace. Default: ``"aitune"``.
        component: Component name within the namespace. Default: ``"backend"``.
        endpoint: Endpoint name within the component. Default: ``"generate"``.
        enable_nats: Enable NATS JetStream for KV cache events. Default: ``False``.
        model_name: Name advertised to the Dynamo frontend. Defaults to ``model_path``.
    """

    type: Literal["image", "video", "embedding"]
    model_path: str
    mapping: Callable | None = None
    namespace: str = "aitune"
    component: str = "backend"
    endpoint: str = "generate"
    enable_nats: bool = False
    model_name: str | None = None


# ---------------------------------------------------------------------------
# Valid type set (internal)
# ---------------------------------------------------------------------------
# Single source of truth for the three P0 modalities AITune supports as Dynamo
# endpoints. Used for validation in both `dynamo_worker` and `_pack_response`.

_VALID_TYPES: frozenset[str] = frozenset({"image", "video", "embedding"})


# ---------------------------------------------------------------------------
# Response auto-packing
# ---------------------------------------------------------------------------
# Converts the raw return value of the user's inference function into the
# wire-format dict that Dynamo expects.  Execution flow:
#   1. If the caller already returned a dict, pass it through unchanged.
#   2. Otherwise dispatch on (result type × modality):
#      - ndarray / Tensor + "embedding"  → OpenAI-style embedding list object
#      - bytes + "image"                 → {"data": [{"b64_json": ...}]}
#      - bytes + "video"                 → NV video response envelope
#      - "audio" is not yet supported — dynamo has no audio_protocol.py
#   3. Any unsupported (result type, modality) combination raises TypeError.


def _pack_response(result: Any, config: DynamoWorkerConfig) -> dict:
    """Convert a user function's return value to the Dynamo wire-format dict.

    Args:
        result: Return value from the user's inference function.
        config: Worker configuration (used for model name and type).

    Returns:
        A dict ready to yield to the Dynamo runtime.

    Raises:
        TypeError: When *result* cannot be converted for *config.type*.
    """
    if isinstance(result, dict):
        return result

    if config.type not in _VALID_TYPES:
        raise TypeError(f"Unsupported modality {config.type!r}. Valid types: {sorted(_VALID_TYPES)}")

    model = config.model_name or config.model_path

    if (isinstance(result, np.ndarray) or isinstance(result, torch.Tensor)) and config.type == "embedding":
        arr = result
        if arr.ndim == 0:
            raise ValueError("Embedding must be a 1D array, got 0D array.")

        if arr.ndim == 1:
            if isinstance(result, torch.Tensor):
                arr = arr.unsqueeze(0)
            else:
                arr = np.expand_dims(arr, 0)

        return {
            "object": "list",
            "data": [{"object": "embedding", "embedding": row.tolist(), "index": i} for i, row in enumerate(arr)],
            "model": model,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    if isinstance(result, bytes):
        b64 = base64.b64encode(result).decode()
        if config.type == "image":
            return {
                "created": int(time.time()),
                "data": [{"b64_json": b64}],
            }
        if config.type == "video":
            return {
                "id": str(uuid4()),
                "object": "video",
                "model": model,
                "status": "completed",
                "progress": 100,
                "created": int(time.time()),
                "data": [{"b64_json": b64}],
            }
        raise TypeError(
            f"bytes response is not supported for modality {config.type!r}. "
            "bytes is only valid for 'image' and 'video'."
        )

    raise TypeError(
        f"Cannot pack response of type {type(result).__name__!r} for modality {config.type!r}. "
        "Return a dict, bytes (image/video), or ndarray/Tensor (embedding)."
    )


def _import_dynamo() -> tuple:
    """Import dynamo runtime dependencies, raising a clear error if absent.

    Returns:
        Tuple of (DistributedRuntime class, dynamo_worker decorator, uvloop module).

    Raises:
        ImportError: When ``ai-dynamo-runtime`` or ``uvloop`` is not installed.
    """
    try:
        import uvloop
        from dynamo.runtime import DistributedRuntime
        from dynamo.runtime import dynamo_worker as _dw

        return DistributedRuntime, _dw, uvloop
    except ImportError as exc:
        raise ImportError("ai-dynamo-runtime is required. Install with: uv pip install 'aitune[dynamo]'") from exc


def _run_dynamo_worker(
    setup: Callable[[], None],
    serve: Callable[[Any], AsyncGenerator[Any, None]],
    namespace: str = "aitune",
    component: str = "backend",
    endpoint: str = "generate",
    enable_nats: bool = False,
    on_ready: Callable[[Any, Any], Coroutine] | None = None,
) -> None:
    """Low-level runtime loop. Call ``setup()`` once, serve until SIGTERM/SIGINT.

    Args:
        setup: Zero-argument initializer called before serving starts.
        serve: Async generator ``serve(request) -> AsyncIterable[response]``.
        namespace: Dynamo service namespace.
        component: Component name within the namespace.
        endpoint: Endpoint name within the component.
        enable_nats: Enable NATS JetStream for KV cache events.
        on_ready: Optional async callable invoked after endpoint registration.
    """
    _, _dw_decorator, uvloop = _import_dynamo()

    logger.info("Setting up model...")
    setup()
    logger.info("Starting Dynamo endpoint %s.%s.%s", namespace, component, endpoint)

    @_dw_decorator(enable_nats=enable_nats)
    async def _worker(runtime: Any) -> None:
        loop = asyncio.get_running_loop()

        async def _shutdown(rt: Any) -> None:
            rt.shutdown()

        _pending_tasks: set = set()

        def _signal_handler() -> None:
            task = loop.create_task(_shutdown(runtime))
            _pending_tasks.add(task)
            task.add_done_callback(_pending_tasks.discard)

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _signal_handler)

        ep = runtime.endpoint(f"{namespace}.{component}.{endpoint}")

        # registration of model must be successful before serving the endpoint
        if on_ready is not None:
            try:
                await on_ready(runtime, ep)
            except Exception:
                logger.exception("on_ready callback failed; shutting down")
                runtime.shutdown()
                raise

        await ep.serve_endpoint(serve)

    uvloop.install()
    asyncio.run(_worker())


class DynamoWorker:
    """Base class for AITune Dynamo workers (power-user API).

    Subclass, override :meth:`setup` and :meth:`serve`, then call :meth:`run`.

    Class Attributes:
        namespace: Dynamo service namespace (default ``"aitune"``).
        component: Component name within the namespace (default ``"backend"``).
        endpoint_name: Endpoint name within the component (default ``"generate"``).

    Example:
        >>> import aitune.dynamo as dyn
        >>> class MyWorker(dyn.DynamoWorker):
        ...     def setup(self):
        ...         pass  # load or tune model here; store on self
        ...     async def serve(self, request):
        ...         yield request  # replace with model call
        >>> # MyWorker().run()  # blocks until SIGTERM/SIGINT
    """

    namespace: str = "aitune"
    component: str = "backend"
    endpoint_name: str = "generate"

    def setup(self) -> None:
        """Initialize the model. Called once before serving starts.

        Raises:
            NotImplementedError: Must be overridden by subclasses.
        """
        raise NotImplementedError

    async def on_ready(self, runtime: Any, endpoint: Any) -> None:
        """Called after the Dynamo endpoint is registered.

        Override to call ``register_model`` or perform post-startup work.

        Args:
            runtime: The :class:`DistributedRuntime` instance.
            endpoint: The registered Dynamo :class:`Endpoint` object.
        """

    async def serve(self, request: Any) -> AsyncGenerator[Any, None]:
        """Handle one request. Async generator — yield response chunks.

        Args:
            request: Incoming request payload.

        Raises:
            NotImplementedError: Must be overridden by subclasses.
        """
        raise NotImplementedError
        yield  # pragma: no cover

    def run(self, enable_nats: bool = False) -> None:
        """Start serving. Blocks until SIGTERM/SIGINT.

        Args:
            enable_nats: Enable NATS JetStream for KV cache events.
        """
        on_ready = None if type(self).on_ready is DynamoWorker.on_ready else self.on_ready
        _run_dynamo_worker(
            setup=self.setup,
            serve=self.serve,
            namespace=self.namespace,
            component=self.component,
            endpoint=self.endpoint_name,
            enable_nats=enable_nats,
            on_ready=on_ready,
        )


# ---------------------------------------------------------------------------
# Dynamo wiring table (internal)
# ---------------------------------------------------------------------------
# Maps each modality string to the three Dynamo objects needed to register a
# model: a ModelInput enum, a ModelType enum, and the Pydantic request class.
# All imports are deferred so the `dynamo` optional dependency is only required
# at call time, not at module import.


def _get_wiring(type_: str) -> tuple:
    """Return (ModelInput, ModelType, request_class) for *type_*.

    All imports are deferred because ``dynamo`` is optional.

    Args:
        type_: Modality type string (``"image"``, ``"video"``, ``"embedding"``).

    Returns:
        Tuple of (ModelInput enum value, ModelType enum value, request Pydantic class).
    """
    from dynamo.llm import ModelInput, ModelType
    from pydantic import BaseModel

    class EmbeddingRequest(BaseModel):
        model: str
        input: str | list[str] | list[list[str]]
        user: str | None = None
        dimensions: int | None = None  # only supported in text-embedding-3 and later models from OpenAI

    from dynamo.common.protocols.image_protocol import NvCreateImageRequest
    from dynamo.common.protocols.video_protocol import NvCreateVideoRequest
    # "audio" is absent: dynamo defines ModelType.Audios and RequestType.AUDIO_GENERATION
    # but audio_protocol.py (NvCreateAudioRequest) does not yet exist upstream.

    table = {
        "embedding": (ModelInput.Text, ModelType.Embedding, EmbeddingRequest),
        "image": (ModelInput.Text, ModelType.Images, NvCreateImageRequest),
        "video": (ModelInput.Text, ModelType.Videos, NvCreateVideoRequest),
    }
    return table[type_]


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------
# `_HighLevelDynamoWorker` is the internal DynamoWorker subclass built from a
# DynamoWorkerConfig.  `dynamo_worker()` is the public function users call:
# it validates the config, constructs the worker, and blocks until shutdown.


class _HighLevelDynamoWorker(DynamoWorker):
    """Internal worker built from a DynamoWorkerConfig. Not part of public API."""

    def __init__(
        self,
        model_or_fn: nn.Module | Callable,
        config: DynamoWorkerConfig,
    ) -> None:
        self._model_or_fn = model_or_fn
        self._config = config
        self.namespace = config.namespace
        self.component = config.component
        self.endpoint_name = config.endpoint
        self._request_cls = None

    def setup(self) -> None:
        """No-op: the model is already initialized before dynamo_worker() is called."""

    async def on_ready(self, runtime: Any, endpoint: Any) -> None:
        """Register this worker with the Dynamo frontend."""
        from dynamo.llm import register_model

        model_input, model_type, request_cls = _get_wiring(self._config.type)
        self._request_cls = request_cls  # cache for serve()
        served_name = self._config.model_name or self._config.model_path
        await register_model(
            model_input,
            model_type,
            endpoint,
            self._config.model_path,
            model_name=served_name,
            **self._version_specific_register_model_kwargs(),
        )
        logger.info(
            "Registered '%s' with Dynamo frontend as %s model",
            served_name,
            self._config.type,
        )

    async def serve(self, request: Any) -> AsyncGenerator[Any, None]:
        """Deserialize request, run user function in executor, pack and yield response."""
        loop = asyncio.get_running_loop()
        if self._request_cls is not None:
            request_cls = self._request_cls
        else:
            _, _, request_cls = _get_wiring(self._config.type)

        # Deserialize: Dynamo sends a dict; request_cls is a Pydantic model.
        typed_req = request_cls(**request) if isinstance(request, dict) else request

        if self._config.mapping is not None:
            kwargs = self._config.mapping(typed_req)
            if not isinstance(kwargs, dict):
                raise TypeError(f"DynamoWorkerConfig.mapping must return a dict, got {type(kwargs).__name__!r}.")
            result = await loop.run_in_executor(None, lambda: self._model_or_fn(**kwargs))
        else:
            result = await loop.run_in_executor(None, lambda: self._model_or_fn(typed_req))

        yield _pack_response(result, self._config)

    def _version_specific_register_model_kwargs(self) -> dict:
        """From version 1.3.0 dynamo introduces a new WorkerType enum and made it required."""
        from dynamo.common import __version__ as dynamo_version
        from packaging.version import Version

        if Version(dynamo_version) >= Version("1.3.0"):
            from dynamo.llm import WorkerType

            return {"worker_type": WorkerType.Aggregated}
        else:
            return {}


def dynamo_worker(
    model_or_fn: nn.Module | Callable,
    config: DynamoWorkerConfig,
) -> None:
    """Serve a tuned model as a Dynamo worker endpoint.

    The minimal path to serving after ``ait.tune()`` or ``ait.load()``:

    1. Build a :class:`DynamoWorkerConfig` for your modality.
    2. Call this function. It blocks until SIGTERM/SIGINT.

    Startup validation happens before any Dynamo runtime is started.
    Modality-specific request deserialization, ``register_model``, sync-to-async
    wrapping, and response packing are handled automatically.

    Args:
        model_or_fn: A ``torch.nn.Module`` (requires ``config.mapping``) or any
            callable. If callable and ``config.mapping`` is ``None``, the raw
            Dynamo request object is passed as the sole argument.
        config: Worker configuration including modality type, model path, and
            optional request adapter.

    Raises:
        ValueError: If ``config.type`` is not a supported P0 modality, or if
            a ``torch.nn.Module`` is passed without a ``mapping``.
        ImportError: If ``ai-dynamo-runtime`` is not installed.

    Example:
        >>> import aitune.dynamo as dyn
        >>> import numpy as np
        >>> def embed(request):
        ...     return np.zeros((1, 768))  # fixed embedding
        >>> config = dyn.DynamoWorkerConfig(
        ...     type="embedding",
        ...     model_path="org/my-embed-model",
        ... )
        >>> # dyn.dynamo_worker(embed, config)  # blocks until shutdown
    """
    # --- Startup validation ---
    if config.type not in _VALID_TYPES:
        raise ValueError(f"Unknown modality type {config.type!r}. Supported types: {', '.join(sorted(_VALID_TYPES))}.")

    if isinstance(model_or_fn, nn.Module) and config.mapping is None:
        raise ValueError(
            "A mapping function is required when passing a torch.nn.Module. Provide config.mapping=fn(request) -> dict."
        )

    worker = _HighLevelDynamoWorker(model_or_fn, config)
    worker.run(enable_nats=config.enable_nats)
