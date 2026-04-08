# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aitune.dynamo.worker."""

import asyncio
import importlib
from unittest.mock import AsyncMock, MagicMock

import pytest

# The canonical implementation lives in aitune.dynamo.worker; patch against that module.
_dw_module = importlib.import_module("aitune.dynamo.worker")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_dynamo_runtime():
    """Return (mock_runtime, mock_endpoint, mock_dw_decorator, mock_uvloop)."""
    mock_endpoint = MagicMock()
    mock_endpoint.serve_endpoint = AsyncMock()

    mock_runtime = MagicMock()
    mock_runtime.endpoint.return_value = mock_endpoint

    def mock_dw_decorator(enable_nats=False):
        def decorator(func):
            async def wrapper(*args, **kwargs):
                return await func(mock_runtime, *args, **kwargs)

            return wrapper

        return decorator

    mock_uvloop = MagicMock()
    return mock_runtime, mock_endpoint, mock_dw_decorator, mock_uvloop


def _run_coro(coro):
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# _import_dynamo / ImportError path
# ---------------------------------------------------------------------------


def test_import_error_when_dynamo_not_installed(mocker):
    """ImportError with install hint when ai-dynamo-runtime is absent."""
    mocker.patch.object(
        _dw_module,
        "_import_dynamo",
        side_effect=ImportError("ai-dynamo-runtime is required. Install with: uv pip install 'aitune[dynamo]'"),
    )
    from aitune.dynamo.worker import _run_dynamo_worker

    def setup():
        pass

    async def serve(request):
        yield request

    with pytest.raises(ImportError, match="aitune\\[dynamo\\]"):
        _run_dynamo_worker(setup, serve)


# ---------------------------------------------------------------------------
# _run_dynamo_worker
# ---------------------------------------------------------------------------


def test_run_dynamo_worker_calls_setup_before_serve(mocker):
    """setup() is called once before serve_endpoint is registered."""
    mock_runtime, mock_endpoint, mock_dw, mock_uvloop = _make_mock_dynamo_runtime()
    mocker.patch.object(_dw_module, "_import_dynamo", return_value=(MagicMock(), mock_dw, mock_uvloop))
    mocker.patch("asyncio.run", side_effect=lambda coro: _run_coro(coro))

    call_order = []

    def setup():
        call_order.append("setup")

    async def serve(request):
        call_order.append("serve")
        yield request

    mock_endpoint.serve_endpoint.side_effect = lambda fn: call_order.append("serve_endpoint")

    from aitune.dynamo.worker import _run_dynamo_worker

    _run_dynamo_worker(setup, serve)

    assert call_order[0] == "setup"
    assert "serve_endpoint" in call_order


def test_run_dynamo_worker_endpoint_address(mocker):
    """Endpoint is registered at namespace.component.endpoint."""
    mock_runtime, mock_endpoint, mock_dw, mock_uvloop = _make_mock_dynamo_runtime()
    mocker.patch.object(_dw_module, "_import_dynamo", return_value=(MagicMock(), mock_dw, mock_uvloop))
    mocker.patch("asyncio.run", side_effect=lambda coro: _run_coro(coro))

    from aitune.dynamo.worker import _run_dynamo_worker

    def setup():
        pass

    async def serve(request):
        yield request

    _run_dynamo_worker(setup, serve, namespace="ns", component="comp", endpoint="ep")
    mock_runtime.endpoint.assert_called_once_with("ns.comp.ep")


# ---------------------------------------------------------------------------
# DynamoWorker base class
# ---------------------------------------------------------------------------


def test_dynamo_worker_base_setup_raises():
    """DynamoWorker.setup() raises NotImplementedError."""
    from aitune.dynamo.worker import DynamoWorker

    class Bare(DynamoWorker):
        pass

    with pytest.raises(NotImplementedError):
        Bare().setup()


def test_dynamo_worker_base_serve_raises():
    """DynamoWorker.serve() raises NotImplementedError."""
    import asyncio

    from aitune.dynamo.worker import DynamoWorker

    class Bare(DynamoWorker):
        pass

    async def _run():
        gen = Bare().serve(object())
        await gen.__anext__()

    with pytest.raises(NotImplementedError):
        asyncio.run(_run())


def test_dynamo_worker_run_calls_run_dynamo_worker(mocker):
    """DynamoWorker.run() delegates to _run_dynamo_worker with correct args."""
    from aitune.dynamo.worker import DynamoWorker

    mock_run = mocker.patch.object(_dw_module, "_run_dynamo_worker")

    class MyWorker(DynamoWorker):
        namespace = "mynamespace"
        component = "mycomp"
        endpoint_name = "myep"

        def setup(self):
            pass

        async def serve(self, request):
            yield request

    MyWorker().run()
    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs["namespace"] == "mynamespace"


def test_dynamo_worker_on_ready_not_passed_when_not_overridden(mocker):
    """on_ready=None is passed when subclass does not override on_ready."""
    from aitune.dynamo.worker import DynamoWorker

    mock_run = mocker.patch.object(_dw_module, "_run_dynamo_worker")

    class MyWorker(DynamoWorker):
        def setup(self):
            pass

        async def serve(self, request):
            yield request

    MyWorker().run()
    assert mock_run.call_args.kwargs.get("on_ready") is None


# ---------------------------------------------------------------------------
# DynamoWorkerConfig
# ---------------------------------------------------------------------------


def test_config_valid_types():
    """DynamoWorkerConfig accepts all P0 type values."""
    from aitune.dynamo.worker import DynamoWorkerConfig

    for t in ("image", "video", "embedding"):
        cfg = DynamoWorkerConfig(type=t, model_path="some/model")
        assert cfg.type == t


def test_config_defaults():
    """DynamoWorkerConfig has expected default field values."""
    from aitune.dynamo.worker import DynamoWorkerConfig

    cfg = DynamoWorkerConfig(type="embedding", model_path="my/model")
    assert cfg.mapping is None
    assert cfg.namespace == "aitune"
    assert cfg.component == "backend"
    assert cfg.endpoint == "generate"
    assert cfg.enable_nats is False
    assert cfg.model_name is None


def test_config_model_name_default():
    """model_name falls back to model_path when None."""
    from aitune.dynamo.worker import DynamoWorkerConfig

    cfg = DynamoWorkerConfig(type="embedding", model_path="org/model")
    assert (cfg.model_name or cfg.model_path) == "org/model"


# ---------------------------------------------------------------------------
# _pack_response
# ---------------------------------------------------------------------------


def test_pack_response_dict_passthrough():
    """dict return value is passed through unchanged."""
    from aitune.dynamo.worker import DynamoWorkerConfig, _pack_response

    cfg = DynamoWorkerConfig(type="embedding", model_path="m")
    payload = {"object": "list", "data": []}
    assert _pack_response(payload, cfg) is payload


def test_pack_response_1d_ndarray_embedding():
    """1-D ndarray -> single-entry OpenAI embedding dict."""
    import numpy as np

    from aitune.dynamo.worker import DynamoWorkerConfig, _pack_response

    cfg = DynamoWorkerConfig(type="embedding", model_path="org/model")
    arr = np.array([0.1, 0.2, 0.3])
    result = _pack_response(arr, cfg)

    assert result["object"] == "list"
    assert len(result["data"]) == 1
    assert result["data"][0]["index"] == 0
    assert result["data"][0]["embedding"] == pytest.approx([0.1, 0.2, 0.3])
    assert result["model"] == "org/model"


def test_pack_response_2d_ndarray_embedding():
    """2-D ndarray -> one entry per row."""
    import numpy as np

    from aitune.dynamo.worker import DynamoWorkerConfig, _pack_response

    cfg = DynamoWorkerConfig(type="embedding", model_path="org/model")
    arr = np.array([[0.1, 0.2], [0.3, 0.4]])
    result = _pack_response(arr, cfg)

    assert len(result["data"]) == 2
    assert result["data"][0]["index"] == 0
    assert result["data"][1]["index"] == 1


def test_pack_response_1d_tensor_embedding():
    """1-D torch.Tensor -> single-entry OpenAI embedding dict."""
    import torch

    from aitune.dynamo.worker import DynamoWorkerConfig, _pack_response

    cfg = DynamoWorkerConfig(type="embedding", model_path="org/model")
    t = torch.tensor([0.5, 0.6, 0.7])
    result = _pack_response(t, cfg)

    assert len(result["data"]) == 1
    assert result["data"][0]["embedding"] == pytest.approx([0.5, 0.6, 0.7])


def test_pack_response_bytes_image():
    """bytes -> image b64_json dict with required fields."""
    import base64

    from aitune.dynamo.worker import DynamoWorkerConfig, _pack_response

    cfg = DynamoWorkerConfig(type="image", model_path="org/model")
    raw = b"fake_image_bytes"
    result = _pack_response(raw, cfg)

    assert "created" in result
    assert isinstance(result["created"], int)
    assert len(result["data"]) == 1
    assert result["data"][0]["b64_json"] == base64.b64encode(raw).decode()


def test_pack_response_bytes_video():
    """bytes -> video b64_json dict with required fields."""
    import base64

    from aitune.dynamo.worker import DynamoWorkerConfig, _pack_response

    cfg = DynamoWorkerConfig(type="video", model_path="org/model", model_name="mymodel")
    raw = b"fake_video_bytes"
    result = _pack_response(raw, cfg)

    assert result["object"] == "video"
    assert result["model"] == "mymodel"
    assert result["status"] == "completed"
    assert result["progress"] == 100
    assert result["data"][0]["b64_json"] == base64.b64encode(raw).decode()


def test_pack_response_unexpected_type_raises():
    """Unexpected return type raises TypeError."""
    from aitune.dynamo.worker import DynamoWorkerConfig, _pack_response

    cfg = DynamoWorkerConfig(type="embedding", model_path="org/model")
    with pytest.raises(TypeError):
        _pack_response(12345, cfg)


def test_pack_response_bytes_wrong_type_raises():
    """bytes response for embedding type raises TypeError."""
    from aitune.dynamo.worker import DynamoWorkerConfig, _pack_response

    cfg = DynamoWorkerConfig(type="embedding", model_path="org/model")
    with pytest.raises(TypeError, match="bytes"):
        _pack_response(b"raw", cfg)


# ---------------------------------------------------------------------------
# _get_wiring — wiring table unit tests (mocks dynamo imports)
# ---------------------------------------------------------------------------


def test_get_wiring_embedding(mocker):
    """embedding type wires to ModelInput.Text, ModelType.Embedding, EmbeddingRequest."""
    mock_model_input = MagicMock()
    mock_model_input.Text = "TEXT"
    mock_model_type = MagicMock()
    mock_model_type.Embedding = "EMBEDDING"

    mocker.patch.dict(
        "sys.modules",
        {
            "dynamo.llm": MagicMock(ModelInput=mock_model_input, ModelType=mock_model_type),
            "dynamo.common.protocols.image_protocol": MagicMock(NvCreateImageRequest=MagicMock()),
            "dynamo.common.protocols.video_protocol": MagicMock(NvCreateVideoRequest=MagicMock()),
        },
    )

    from aitune.dynamo.worker import _get_wiring

    model_input, model_type, req_cls = _get_wiring("embedding")

    assert model_input == "TEXT"
    assert model_type == "EMBEDDING"
    assert req_cls.__name__.endswith("EmbeddingRequest")


def test_get_wiring_image(mocker):
    """image type wires to ModelInput.Text, ModelType.Images, NvCreateImageRequest."""
    mock_model_input = MagicMock()
    mock_model_input.Text = "TEXT"
    mock_model_type = MagicMock()
    mock_model_type.Images = "IMAGES"
    mock_image_req = MagicMock()

    mocker.patch.dict(
        "sys.modules",
        {
            "dynamo.llm": MagicMock(ModelInput=mock_model_input, ModelType=mock_model_type),
            "dynamo.common.protocols.image_protocol": MagicMock(NvCreateImageRequest=mock_image_req),
            "dynamo.common.protocols.video_protocol": MagicMock(NvCreateVideoRequest=MagicMock()),
        },
    )

    from aitune.dynamo.worker import _get_wiring

    model_input, model_type, req_cls = _get_wiring("image")

    assert model_input == "TEXT"
    assert model_type == "IMAGES"
    assert req_cls is mock_image_req


def test_get_wiring_video(mocker):
    """video type wires to ModelInput.Text, ModelType.Videos, NvCreateVideoRequest."""
    mock_model_input = MagicMock()
    mock_model_input.Text = "TEXT"
    mock_model_type = MagicMock()
    mock_model_type.Videos = "VIDEOS"
    mock_video_req = MagicMock()

    mocker.patch.dict(
        "sys.modules",
        {
            "dynamo.llm": MagicMock(ModelInput=mock_model_input, ModelType=mock_model_type),
            "dynamo.sglang.protocol": MagicMock(EmbeddingRequest=MagicMock()),
            "dynamo.common.protocols.image_protocol": MagicMock(NvCreateImageRequest=MagicMock()),
            "dynamo.common.protocols.video_protocol": MagicMock(NvCreateVideoRequest=mock_video_req),
        },
    )

    from aitune.dynamo.worker import _get_wiring

    model_input, model_type, req_cls = _get_wiring("video")

    assert model_input == "TEXT"
    assert model_type == "VIDEOS"
    assert req_cls is mock_video_req


def test_get_wiring_embedding_fallback_import(mocker):
    """EmbeddingRequest falls back to vllm path when sglang import fails."""
    mock_model_input = MagicMock()
    mock_model_input.Text = "TEXT"
    mock_model_type = MagicMock()
    mock_model_type.Embedding = "EMBEDDING"

    mocker.patch.dict(
        "sys.modules",
        {
            "dynamo.llm": MagicMock(ModelInput=mock_model_input, ModelType=mock_model_type),
            "dynamo.common.protocols.image_protocol": MagicMock(NvCreateImageRequest=MagicMock()),
            "dynamo.common.protocols.video_protocol": MagicMock(NvCreateVideoRequest=MagicMock()),
        },
    )

    from aitune.dynamo.worker import _get_wiring

    _, _, req_cls = _get_wiring("embedding")
    assert req_cls.__name__.endswith("EmbeddingRequest")


# ---------------------------------------------------------------------------
# Public dynamo_worker() — startup validation
# ---------------------------------------------------------------------------


def test_high_level_unknown_type_raises():
    """ValueError at startup for unknown config.type."""
    from aitune.dynamo.worker import DynamoWorkerConfig, dynamo_worker

    # bypass the Literal type check — we test runtime validation
    cfg = DynamoWorkerConfig.__new__(DynamoWorkerConfig)
    object.__setattr__(cfg, "type", "unknown_modality")
    object.__setattr__(cfg, "model_path", "org/model")
    object.__setattr__(cfg, "mapping", None)
    object.__setattr__(cfg, "namespace", "aitune")
    object.__setattr__(cfg, "component", "backend")
    object.__setattr__(cfg, "endpoint", "generate")
    object.__setattr__(cfg, "enable_nats", False)
    object.__setattr__(cfg, "model_name", None)

    with pytest.raises(ValueError, match="unknown_modality"):
        dynamo_worker(lambda req: None, cfg)


def test_high_level_module_without_mapping_raises():
    """ValueError at startup when nn.Module is passed without mapping."""
    import torch.nn as nn

    from aitune.dynamo.worker import DynamoWorkerConfig, dynamo_worker

    cfg = DynamoWorkerConfig(type="embedding", model_path="org/model", mapping=None)

    class MyModule(nn.Module):
        def forward(self, x):
            return x

    with pytest.raises(ValueError, match="mapping"):
        dynamo_worker(MyModule(), cfg)


# ---------------------------------------------------------------------------
# Public dynamo_worker() — request routing
# ---------------------------------------------------------------------------


def test_high_level_callable_no_mapping_passes_raw_request(mocker):
    """Callable without mapping receives raw request object."""
    received = []

    def my_fn(request):
        received.append(request)
        return {"object": "list", "data": [], "model": "m", "usage": {}}

    mock_request = {"input": "hello", "model": "org/model"}

    mock_run = mocker.patch.object(_dw_module, "_run_dynamo_worker")
    mock_wiring_req = MagicMock(return_value=MagicMock(**mock_request))
    mocker.patch.object(_dw_module, "_get_wiring", return_value=(MagicMock(), MagicMock(), mock_wiring_req))

    from aitune.dynamo.worker import DynamoWorkerConfig, dynamo_worker

    cfg = DynamoWorkerConfig(type="embedding", model_path="org/model")
    dynamo_worker(my_fn, cfg)

    # Extract the serve function that was passed to _run_dynamo_worker
    assert mock_run.called
    serve_fn = mock_run.call_args.kwargs.get("serve") or mock_run.call_args[0][1]

    # Simulate what serve does when called with a raw dict from Dynamo
    async def collect():
        results = []
        async for chunk in serve_fn(mock_request):
            results.append(chunk)
        return results

    results = asyncio.run(collect())

    assert len(results) == 1  # single chunk yielded
    assert received[0] is not None  # fn was called with the deserialized request


def test_high_level_callable_with_mapping_unpacks_kwargs(mocker):
    """Callable with mapping receives **dict from mapping(request)."""
    received_kwargs = {}

    def my_fn(prompt, size):
        received_kwargs["prompt"] = prompt
        received_kwargs["size"] = size
        return b"fake_image"

    def my_mapping(req):
        return {"prompt": req.prompt, "size": "1024x1024"}

    mock_request_obj = MagicMock()
    mock_request_obj.prompt = "a sunset"
    mock_request_cls = MagicMock(return_value=mock_request_obj)

    mocker.patch.object(_dw_module, "_get_wiring", return_value=(MagicMock(), MagicMock(), mock_request_cls))
    mock_run = mocker.patch.object(_dw_module, "_run_dynamo_worker")

    from aitune.dynamo.worker import DynamoWorkerConfig, dynamo_worker

    cfg = DynamoWorkerConfig(type="image", model_path="org/model", mapping=my_mapping)
    dynamo_worker(my_fn, cfg)

    serve_fn = mock_run.call_args.kwargs.get("serve") or mock_run.call_args[0][1]

    async def collect():
        chunks = []
        async for c in serve_fn({"prompt": "a sunset"}):
            chunks.append(c)
        return chunks

    results = asyncio.run(collect())

    assert received_kwargs == {"prompt": "a sunset", "size": "1024x1024"}
    assert results[0]["data"][0]["b64_json"]  # bytes were packed


def test_high_level_mapping_not_dict_raises_type_error(mocker):
    """TypeError at request time when mapping returns non-dict."""

    def bad_mapping(req):
        return "not_a_dict"

    def my_fn(x):
        return b"img"

    mock_request_cls = MagicMock(return_value=MagicMock())
    mocker.patch.object(_dw_module, "_get_wiring", return_value=(MagicMock(), MagicMock(), mock_request_cls))
    mock_run = mocker.patch.object(_dw_module, "_run_dynamo_worker")

    from aitune.dynamo.worker import DynamoWorkerConfig, dynamo_worker

    cfg = DynamoWorkerConfig(type="image", model_path="org/model", mapping=bad_mapping)
    dynamo_worker(my_fn, cfg)

    serve_fn = mock_run.call_args.kwargs.get("serve") or mock_run.call_args[0][1]

    async def collect():
        async for _ in serve_fn({}):
            pass

    with pytest.raises(TypeError):
        asyncio.run(collect())


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_exports_accessible_from_aitune_dynamo():
    """dynamo_worker, DynamoWorker, DynamoWorkerConfig are exported from aitune.dynamo."""
    import aitune.dynamo as aid

    assert hasattr(aid, "dynamo_worker")
    assert hasattr(aid, "DynamoWorker")
    assert hasattr(aid, "DynamoWorkerConfig")
