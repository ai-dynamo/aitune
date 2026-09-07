# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the AITune Dynamo integration."""

import asyncio
import importlib
from unittest.mock import AsyncMock, MagicMock, call

import pytest

_dw_module = importlib.import_module("aitune.dynamo.worker")
_dd_module = importlib.import_module("aitune.dynamo.distributed")
_dp_module = importlib.import_module("aitune.dynamo.protocol")
_dr_module = importlib.import_module("aitune.dynamo.runtime")

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


def _pack_response(result, config):
    model_name = config.model_name or config.model_path
    return _dp_module.pack_response(result, config.type, model_name)


# ---------------------------------------------------------------------------
# _import_dynamo / ImportError path
# ---------------------------------------------------------------------------


def test_import_error_when_dynamo_not_installed(mocker):
    """ImportError with install hint when ai-dynamo-runtime is absent."""
    mocker.patch.object(
        _dr_module,
        "_import_dynamo",
        side_effect=ImportError("ai-dynamo-runtime is required. Install with: uv pip install 'aitune[dynamo]'"),
    )
    from aitune.dynamo.runtime import run_dynamo_worker

    def setup():
        pass

    async def serve(request):
        yield request

    with pytest.raises(ImportError, match="aitune\\[dynamo\\]"):
        run_dynamo_worker(setup, serve)


def test_import_dynamo_loads_optional_runtime(mocker):
    runtime = MagicMock(dynamo_worker=MagicMock())
    uvloop = MagicMock()
    mocker.patch.dict("sys.modules", {"dynamo.runtime": runtime, "uvloop": uvloop})

    assert _dr_module._import_dynamo() == (runtime.dynamo_worker, uvloop)


def test_import_dynamo_reports_missing_optional_runtime(mocker):
    mocker.patch.dict("sys.modules", {"uvloop": None})

    with pytest.raises(ImportError, match=r"aitune\[dynamo\]"):
        _dr_module._import_dynamo()


# ---------------------------------------------------------------------------
# run_dynamo_worker
# ---------------------------------------------------------------------------


def test_run_dynamo_worker_calls_setup_and_warmup_before_serve(mocker):
    """Setup and warmup run before serve_endpoint is registered."""
    _, mock_endpoint, mock_dw, mock_uvloop = _make_mock_dynamo_runtime()
    mocker.patch.object(_dr_module, "_import_dynamo", return_value=(mock_dw, mock_uvloop))
    mocker.patch("asyncio.run", side_effect=lambda coro: _run_coro(coro))

    call_order = []

    def setup():
        call_order.append("setup")

    def warmup():
        call_order.append("warmup")

    async def serve(request):
        call_order.append("serve")
        yield request

    mock_endpoint.serve_endpoint.side_effect = lambda fn: call_order.append("serve_endpoint")

    from aitune.dynamo.runtime import run_dynamo_worker

    run_dynamo_worker(setup, serve, warmup=warmup)

    assert call_order == ["setup", "warmup", "serve_endpoint"]


def test_run_dynamo_worker_endpoint_address(mocker):
    """Endpoint is registered at namespace.component.endpoint."""
    mock_runtime, _, mock_dw, mock_uvloop = _make_mock_dynamo_runtime()
    mocker.patch.object(_dr_module, "_import_dynamo", return_value=(mock_dw, mock_uvloop))
    mocker.patch("asyncio.run", side_effect=lambda coro: _run_coro(coro))

    from aitune.dynamo.runtime import run_dynamo_worker

    def setup():
        pass

    async def serve(request):
        yield request

    run_dynamo_worker(setup, serve, namespace="ns", component="comp", endpoint="ep")
    mock_runtime.endpoint.assert_called_once_with("ns.comp.ep")


def test_run_dynamo_worker_calls_on_ready(mocker):
    mock_runtime, _, mock_dw, mock_uvloop = _make_mock_dynamo_runtime()
    mocker.patch.object(_dr_module, "_import_dynamo", return_value=(mock_dw, mock_uvloop))
    mocker.patch("asyncio.run", side_effect=lambda coro: _run_coro(coro))
    on_ready = AsyncMock()

    async def serve(request):
        yield request

    _dr_module.run_dynamo_worker(lambda: None, serve, on_ready=on_ready)

    on_ready.assert_awaited_once_with(mock_runtime, mock_runtime.endpoint.return_value)


def test_run_dynamo_worker_shuts_down_when_on_ready_fails(mocker):
    mock_runtime, _, mock_dw, mock_uvloop = _make_mock_dynamo_runtime()
    mocker.patch.object(_dr_module, "_import_dynamo", return_value=(mock_dw, mock_uvloop))
    mocker.patch("asyncio.run", side_effect=lambda coro: _run_coro(coro))
    on_ready = AsyncMock(side_effect=RuntimeError("registration failed"))

    async def serve(request):
        yield request

    with pytest.raises(RuntimeError, match="registration failed"):
        _dr_module.run_dynamo_worker(lambda: None, serve, on_ready=on_ready)

    mock_runtime.shutdown.assert_called_once()


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
    """DynamoWorker.run() delegates to run_dynamo_worker with correct args."""
    from aitune.dynamo.worker import DynamoWorker

    mock_run = mocker.patch.object(_dw_module, "run_dynamo_worker")

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

    mock_run = mocker.patch.object(_dw_module, "run_dynamo_worker")

    class MyWorker(DynamoWorker):
        def setup(self):
            pass

        async def serve(self, request):
            yield request

    MyWorker().run()
    assert mock_run.call_args.kwargs.get("on_ready") is None


@pytest.mark.parametrize(
    "initialized,world_size,expected",
    [
        (False, 1, False),
        (True, 1, False),
        (True, 4, True),
    ],
)
def test_should_run_collectively_detects_multi_rank_process_group(mocker, initialized, world_size, expected):
    mocker.patch.object(_dd_module.dist, "is_available", return_value=True)
    mocker.patch.object(_dd_module.dist, "is_initialized", return_value=initialized)
    mocker.patch.object(_dd_module.dist, "get_world_size", return_value=world_size)

    assert _dd_module.should_run_collectively() is expected


def test_should_run_collectively_handles_unavailable_distributed_package(mocker):
    mocker.patch.object(_dd_module.dist, "is_available", return_value=False)
    initialized = mocker.patch.object(_dd_module.dist, "is_initialized")

    assert not _dd_module.should_run_collectively()
    initialized.assert_not_called()


@pytest.mark.parametrize("initialized,rank,expected", [(False, 3, True), (True, 0, True), (True, 1, False)])
def test_is_rank_zero(mocker, initialized, rank, expected):
    mocker.patch.object(_dd_module.dist, "is_initialized", return_value=initialized)
    get_rank = mocker.patch.object(_dd_module.dist, "get_rank", return_value=rank)

    assert _dd_module.is_rank_zero() is expected
    assert get_rank.called is initialized


def test_dynamo_worker_run_uses_collective_runner(mocker):
    from aitune.dynamo.worker import DynamoWorker

    local_runner = mocker.patch.object(_dw_module, "run_dynamo_worker")
    collective_runner = mocker.patch.object(_dw_module, "run_collective_dynamo_worker")
    mocker.patch.object(_dw_module, "should_run_collectively", return_value=True)

    class MyWorker(DynamoWorker):
        def setup(self):
            pass

        async def serve(self, request):
            yield request

    MyWorker().run()

    collective_runner.assert_called_once()
    local_runner.assert_not_called()


def test_run_collective_dynamo_worker_runs_endpoint_on_rank_zero_and_stops_followers(mocker):
    collective_setup = mocker.patch.object(_dd_module, "_run_collective_setup")
    mocker.patch.object(_dd_module.dist, "get_rank", return_value=0)
    run = mocker.patch.object(_dd_module, "run_dynamo_worker", side_effect=RuntimeError("stopped"))
    collective_serve = mocker.patch.object(_dd_module, "_collective_leader_serve", return_value="collective-serve")
    broadcast = mocker.patch.object(_dd_module, "_broadcast_collective_command")

    async def serve(request):
        yield request

    setup = mocker.Mock()
    warmup = mocker.Mock()
    with pytest.raises(RuntimeError, match="stopped"):
        _dd_module.run_collective_dynamo_worker(
            setup,
            serve,
            namespace="ns",
            component="component",
            endpoint="endpoint",
            enable_nats=True,
            on_ready=None,
            warmup=warmup,
        )

    assert collective_setup.call_args_list == [call(setup, stage="setup"), call(warmup, stage="warmup")]
    collective_serve.assert_called_once_with(serve)
    assert run.call_args.kwargs["serve"] == "collective-serve"
    assert run.call_args.kwargs["warmup"] is None
    broadcast.assert_called_once_with(_dd_module._STOP_COMMAND)


def test_run_collective_dynamo_worker_preserves_worker_error_when_stop_broadcast_fails(mocker):
    mocker.patch.object(_dd_module, "_run_collective_setup")
    mocker.patch.object(_dd_module.dist, "get_rank", return_value=0)
    mocker.patch.object(_dd_module, "run_dynamo_worker", side_effect=RuntimeError("worker failed"))
    mocker.patch.object(_dd_module, "_collective_leader_serve", return_value="collective-serve")
    mocker.patch.object(_dd_module, "_broadcast_collective_command", side_effect=RuntimeError("stop failed"))
    log_exception = mocker.patch.object(_dd_module.logger, "exception")

    async def serve(request):
        yield request

    with pytest.raises(RuntimeError, match="worker failed"):
        _dd_module.run_collective_dynamo_worker(
            mocker.Mock(),
            serve,
            namespace="ns",
            component="component",
            endpoint="endpoint",
            enable_nats=True,
            on_ready=None,
        )

    log_exception.assert_called_once_with("Failed to broadcast collective stop command while handling a worker error")


def test_run_collective_dynamo_worker_propagates_stop_broadcast_error(mocker):
    mocker.patch.object(_dd_module, "_run_collective_setup")
    mocker.patch.object(_dd_module.dist, "get_rank", return_value=0)
    mocker.patch.object(_dd_module, "run_dynamo_worker")
    mocker.patch.object(_dd_module, "_collective_leader_serve", return_value="collective-serve")
    mocker.patch.object(_dd_module, "_broadcast_collective_command", side_effect=RuntimeError("stop failed"))
    log_exception = mocker.patch.object(_dd_module.logger, "exception")

    async def serve(request):
        yield request

    with pytest.raises(RuntimeError, match="stop failed"):
        _dd_module.run_collective_dynamo_worker(
            mocker.Mock(),
            serve,
            namespace="ns",
            component="component",
            endpoint="endpoint",
            enable_nats=True,
            on_ready=None,
        )

    log_exception.assert_not_called()


def test_run_collective_dynamo_worker_runs_follower_without_endpoint(mocker):
    collective_setup = mocker.patch.object(_dd_module, "_run_collective_setup")
    mocker.patch.object(_dd_module.dist, "get_rank", return_value=1)
    follower = mocker.patch.object(_dd_module, "_run_collective_follower")
    run = mocker.patch.object(_dd_module, "run_dynamo_worker")

    async def serve(request):
        yield request

    setup = mocker.Mock()
    warmup = mocker.Mock()
    _dd_module.run_collective_dynamo_worker(
        setup,
        serve,
        namespace="ns",
        component="component",
        endpoint="endpoint",
        enable_nats=False,
        on_ready=None,
        warmup=warmup,
    )

    assert collective_setup.call_args_list == [call(setup, stage="setup"), call(warmup, stage="warmup")]
    follower.assert_called_once_with(serve)
    run.assert_not_called()


def test_run_collective_setup_succeeds_when_every_rank_succeeds(mocker):
    setup = mocker.Mock()
    mocker.patch.object(_dd_module, "_collective_errors", return_value=[None, None])

    _dd_module._run_collective_setup(setup)

    setup.assert_called_once()


def test_run_collective_dynamo_worker_does_not_start_after_warmup_failure(mocker):
    setup = mocker.Mock()
    warmup = mocker.Mock()
    collective_setup = mocker.patch.object(
        _dd_module,
        "_run_collective_setup",
        side_effect=[None, RuntimeError("warmup failed")],
    )
    run = mocker.patch.object(_dd_module, "run_dynamo_worker")
    follower = mocker.patch.object(_dd_module, "_run_collective_follower")

    async def serve(request):
        yield request

    with pytest.raises(RuntimeError, match="warmup failed"):
        _dd_module.run_collective_dynamo_worker(
            setup,
            serve,
            namespace="ns",
            component="component",
            endpoint="endpoint",
            enable_nats=False,
            on_ready=None,
            warmup=warmup,
        )

    assert collective_setup.call_args_list == [call(setup, stage="setup"), call(warmup, stage="warmup")]
    run.assert_not_called()
    follower.assert_not_called()


def test_run_collective_setup_preserves_local_failure(mocker):
    error = ValueError("local setup failed")
    mocker.patch.object(_dd_module, "_collective_errors", return_value=["rank 0: ValueError", None])

    with pytest.raises(ValueError, match="local setup failed"):
        _dd_module._run_collective_setup(mocker.Mock(side_effect=error))


def test_run_collective_setup_reports_remote_failure(mocker):
    mocker.patch.object(_dd_module, "_collective_errors", return_value=[None, "rank 1: ValueError: failed"])

    with pytest.raises(RuntimeError, match="rank 1"):
        _dd_module._run_collective_setup(mocker.Mock())


def test_collective_leader_serializes_and_broadcasts_request(mocker):
    broadcast = mocker.patch.object(_dd_module, "_broadcast_collective_command")
    execute = mocker.patch.object(_dd_module, "_execute_collective_request", new_callable=AsyncMock)
    execute.return_value = ["response"]

    async def serve(request):
        yield request

    collective_serve = _dd_module._collective_leader_serve(serve)

    async def collect():
        return [response async for response in collective_serve({"prompt": "hello"})]

    assert asyncio.run(collect()) == ["response"]
    broadcast.assert_called_once_with(_dd_module._RUN_COMMAND, {"prompt": "hello"})
    execute.assert_awaited_once_with(serve, {"prompt": "hello"})


def test_execute_collective_request_reports_remote_failure(mocker):
    mocker.patch.object(_dd_module, "_collective_errors", return_value=[None, "rank 1: RuntimeError: failed"])

    async def serve(request):
        yield request

    with pytest.raises(RuntimeError, match="rank 1"):
        asyncio.run(_dd_module._execute_collective_request(serve, "request"))


def test_execute_collective_request_preserves_local_failure(mocker):
    mocker.patch.object(_dd_module, "_collective_errors", return_value=["rank 0: ValueError: failed", None])

    async def serve(request):
        raise ValueError(request)
        yield

    with pytest.raises(ValueError, match="failed"):
        asyncio.run(_dd_module._execute_collective_request(serve, "failed"))


def test_execute_collective_request_can_discard_follower_responses(mocker):
    mocker.patch.object(_dd_module, "_collective_errors", return_value=[None, None])

    async def serve(request):
        yield request

    assert asyncio.run(_dd_module._execute_collective_request(serve, "response", collect_responses=False)) == []


def test_collective_follower_executes_requests_until_stop(mocker):
    mocker.patch.object(
        _dd_module,
        "_receive_collective_command",
        side_effect=[(_dd_module._RUN_COMMAND, "request"), (_dd_module._STOP_COMMAND, None)],
    )
    execute = mocker.patch.object(_dd_module, "_execute_collective_request", new_callable=AsyncMock)
    mocker.patch.object(_dd_module.signal, "getsignal", return_value=_dd_module.signal.SIG_DFL)
    mocker.patch.object(_dd_module.signal, "signal")

    async def serve(request):
        yield request

    _dd_module._run_collective_follower(serve)

    execute.assert_awaited_once_with(serve, "request", collect_responses=False)


def test_collective_follower_logs_request_failure_and_continues(mocker):
    mocker.patch.object(
        _dd_module,
        "_receive_collective_command",
        side_effect=[(_dd_module._RUN_COMMAND, "request"), (_dd_module._STOP_COMMAND, None)],
    )
    mocker.patch.object(
        _dd_module,
        "_execute_collective_request",
        new_callable=AsyncMock,
        side_effect=RuntimeError("failed"),
    )
    mocker.patch.object(_dd_module.signal, "getsignal", return_value=_dd_module.signal.SIG_DFL)
    mocker.patch.object(_dd_module.signal, "signal")
    log = mocker.patch.object(_dd_module.logger, "exception")

    async def serve(request):
        yield request

    _dd_module._run_collective_follower(serve)

    log.assert_called_once_with("Collective Dynamo request failed")


def test_collective_follower_rejects_unknown_command(mocker):
    mocker.patch.object(_dd_module, "_receive_collective_command", return_value=("invalid", None))
    mocker.patch.object(_dd_module.signal, "getsignal", return_value=_dd_module.signal.SIG_DFL)
    signal = mocker.patch.object(_dd_module.signal, "signal")

    async def serve(request):
        yield request

    with pytest.raises(RuntimeError, match="Unknown collective"):
        _dd_module._run_collective_follower(serve)

    assert signal.call_args_list == [
        call(_dd_module.signal.SIGINT, _dd_module.signal.SIG_IGN),
        call(_dd_module.signal.SIGINT, _dd_module.signal.SIG_DFL),
    ]


@pytest.mark.parametrize("error", [None, ValueError("failed")])
def test_collective_errors_gathers_rank_descriptions(mocker, error):
    mocker.patch.object(_dd_module.dist, "get_rank", return_value=1)
    mocker.patch.object(_dd_module.dist, "get_world_size", return_value=2)

    def gather(errors, local_error):
        errors[:] = [None, local_error]

    mocker.patch.object(_dd_module.dist, "all_gather_object", side_effect=gather)

    errors = _dd_module._collective_errors(error)

    assert errors[0] is None
    expected = None if error is None else "rank 1: ValueError: failed"
    assert errors[1] == expected


def test_broadcast_collective_command(mocker):
    broadcast = mocker.patch.object(_dd_module.dist, "broadcast_object_list")

    _dd_module._broadcast_collective_command(_dd_module._RUN_COMMAND, {"prompt": "hello"})

    broadcast.assert_called_once_with([(_dd_module._RUN_COMMAND, {"prompt": "hello"})], src=0)


def test_receive_collective_command(mocker):
    def broadcast(payload, src):
        assert src == 0
        payload[0] = (_dd_module._RUN_COMMAND, "request")

    mocker.patch.object(_dd_module.dist, "broadcast_object_list", side_effect=broadcast)

    assert _dd_module._receive_collective_command() == (_dd_module._RUN_COMMAND, "request")


def test_receive_collective_command_rejects_empty_message(mocker):
    mocker.patch.object(_dd_module.dist, "broadcast_object_list")

    with pytest.raises(RuntimeError, match="empty collective"):
        _dd_module._receive_collective_command()


# ---------------------------------------------------------------------------
# DynamoWorkerConfig
# ---------------------------------------------------------------------------


def test_config_valid_types():
    """DynamoWorkerConfig accepts all supported type values."""
    from aitune.dynamo.worker import DynamoWorkerConfig

    for t in ("image", "video", "audio", "embedding"):
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
    from aitune.dynamo.worker import DynamoWorkerConfig

    cfg = DynamoWorkerConfig(type="embedding", model_path="m")
    payload = {"object": "list", "data": []}
    assert _pack_response(payload, cfg) is payload


def test_pack_response_1d_ndarray_embedding():
    """1-D ndarray -> single-entry OpenAI embedding dict."""
    import numpy as np

    from aitune.dynamo.worker import DynamoWorkerConfig

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

    from aitune.dynamo.worker import DynamoWorkerConfig

    cfg = DynamoWorkerConfig(type="embedding", model_path="org/model")
    arr = np.array([[0.1, 0.2], [0.3, 0.4]])
    result = _pack_response(arr, cfg)

    assert len(result["data"]) == 2
    assert result["data"][0]["index"] == 0
    assert result["data"][1]["index"] == 1


def test_pack_response_1d_tensor_embedding():
    """1-D torch.Tensor -> single-entry OpenAI embedding dict."""
    import torch

    from aitune.dynamo.worker import DynamoWorkerConfig

    cfg = DynamoWorkerConfig(type="embedding", model_path="org/model")
    t = torch.tensor([0.5, 0.6, 0.7])
    result = _pack_response(t, cfg)

    assert len(result["data"]) == 1
    assert result["data"][0]["embedding"] == pytest.approx([0.5, 0.6, 0.7])


def test_pack_response_bytes_image():
    """bytes -> image b64_json dict with required fields."""
    import base64

    from aitune.dynamo.worker import DynamoWorkerConfig

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

    from aitune.dynamo.worker import DynamoWorkerConfig

    cfg = DynamoWorkerConfig(type="video", model_path="org/model", model_name="mymodel")
    raw = b"fake_video_bytes"
    result = _pack_response(raw, cfg)

    assert result["object"] == "video"
    assert result["model"] == "mymodel"
    assert result["status"] == "completed"
    assert result["progress"] == 100
    assert result["data"][0]["b64_json"] == base64.b64encode(raw).decode()
    assert result["data"][0]["output_format"] == "mp4"


def test_pack_response_bytes_audio():
    """bytes -> audio speech b64_json dict with required fields."""
    import base64

    from aitune.dynamo.worker import DynamoWorkerConfig

    cfg = DynamoWorkerConfig(type="audio", model_path="org/model", model_name="mymodel")
    raw = b"fake_wav_bytes"
    result = _pack_response(raw, cfg)

    assert result["object"] == "audio.speech"
    assert result["model"] == "mymodel"
    assert result["status"] == "completed"
    assert result["progress"] == 100
    assert result["data"][0]["b64_json"] == base64.b64encode(raw).decode()
    assert result["data"][0]["output_format"] == "wav"


def test_pack_response_unexpected_type_raises():
    """Unexpected return type raises TypeError."""
    from aitune.dynamo.worker import DynamoWorkerConfig

    cfg = DynamoWorkerConfig(type="embedding", model_path="org/model")
    with pytest.raises(TypeError):
        _pack_response(12345, cfg)


def test_pack_response_rejects_unknown_modality():
    with pytest.raises(TypeError, match="Unsupported modality"):
        _dp_module.pack_response(12345, "text", "org/model")


def test_pack_response_rejects_scalar_embedding():
    import numpy as np

    with pytest.raises(ValueError, match="0D"):
        _dp_module.pack_response(np.array(1.0), "embedding", "org/model")


@pytest.mark.parametrize("embedding", [pytest.param("numpy", id="numpy"), pytest.param("torch", id="torch")])
def test_pack_response_rejects_embedding_with_more_than_two_dimensions(embedding):
    import numpy as np
    import torch

    result = np.zeros((1, 2, 3)) if embedding == "numpy" else torch.zeros((1, 2, 3))
    with pytest.raises(ValueError, match="1D or 2D"):
        _dp_module.pack_response(result, "embedding", "org/model")


def test_pack_response_bytes_wrong_type_raises():
    """bytes response for embedding type raises TypeError."""
    from aitune.dynamo.worker import DynamoWorkerConfig

    cfg = DynamoWorkerConfig(type="embedding", model_path="org/model")
    with pytest.raises(TypeError, match="bytes"):
        _pack_response(b"raw", cfg)


# ---------------------------------------------------------------------------
# get_wiring — wiring table unit tests (mocks dynamo imports)
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

    from aitune.dynamo.protocol import get_wiring

    model_input, model_type, req_cls = get_wiring("embedding")

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

    from aitune.dynamo.protocol import get_wiring

    model_input, model_type, req_cls = get_wiring("image")

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

    from aitune.dynamo.protocol import get_wiring

    model_input, model_type, req_cls = get_wiring("video")

    assert model_input == "TEXT"
    assert model_type == "VIDEOS"
    assert req_cls is mock_video_req


def test_get_wiring_audio(mocker):
    """audio type wires to ModelInput.Text, ModelType.Audios, NvCreateAudioSpeechRequest."""
    mock_model_input = MagicMock()
    mock_model_input.Text = "TEXT"
    mock_model_type = MagicMock()
    mock_model_type.Audios = "AUDIOS"
    mock_audio_req = MagicMock()

    mocker.patch.dict(
        "sys.modules",
        {
            "dynamo.llm": MagicMock(ModelInput=mock_model_input, ModelType=mock_model_type),
            "dynamo.common.protocols.audio_protocol": MagicMock(NvCreateAudioSpeechRequest=mock_audio_req),
        },
    )

    from aitune.dynamo.protocol import get_wiring

    model_input, model_type, req_cls = get_wiring("audio")

    assert model_input == "TEXT"
    assert model_type == "AUDIOS"
    assert req_cls is mock_audio_req


def test_get_wiring_audio_requires_supported_dynamo(mocker):
    """audio type reports its minimum Dynamo runtime version."""
    mock_model_input = MagicMock(Text="TEXT")
    mock_model_type = MagicMock(spec=[])
    mocker.patch.dict(
        "sys.modules",
        {
            "dynamo.llm": MagicMock(ModelInput=mock_model_input, ModelType=mock_model_type),
            "dynamo.common.protocols.audio_protocol": None,
        },
    )

    with pytest.raises(ImportError, match=r"ai-dynamo-runtime>=1\.1\.0"):
        _dp_module.get_wiring("audio")


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

    from aitune.dynamo.protocol import get_wiring

    _, _, req_cls = get_wiring("embedding")
    assert req_cls.__name__.endswith("EmbeddingRequest")


@pytest.mark.parametrize("version,expected", [("1.2.0", {}), ("1.3.0", {"worker_type": "AGGREGATED"})])
def test_get_register_model_kwargs(mocker, version, expected):
    worker_type = MagicMock(Aggregated="AGGREGATED")
    mocker.patch.dict(
        "sys.modules",
        {
            "dynamo.common": MagicMock(__version__=version),
            "dynamo.llm": MagicMock(WorkerType=worker_type),
        },
    )

    assert _dp_module.get_register_model_kwargs() == expected


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


def test_high_level_setup_caches_protocol_wiring(mocker):
    wiring = (MagicMock(), MagicMock(), MagicMock())
    get_wiring = mocker.patch.object(_dw_module, "get_wiring", return_value=wiring)
    config = _dw_module.DynamoWorkerConfig(type="image", model_path="org/model")
    worker = _dw_module._HighLevelDynamoWorker(MagicMock(), config)

    worker.setup()

    assert worker._wiring == wiring
    get_wiring.assert_called_once_with("image")


def test_high_level_runs_configured_setup_and_warmup(mocker):
    setup = mocker.Mock()
    warmup = mocker.Mock()
    mocker.patch.object(_dw_module, "get_wiring", return_value=(MagicMock(), MagicMock(), MagicMock()))
    config = _dw_module.DynamoWorkerConfig(type="image", model_path="org/model")
    worker = _dw_module._HighLevelDynamoWorker(MagicMock(), config, setup=setup, warmup=warmup)

    worker.setup()
    worker.warmup()

    setup.assert_called_once_with()
    warmup.assert_called_once_with()


@pytest.mark.parametrize("cached_wiring", [False, True])
def test_high_level_on_ready_registers_model(mocker, cached_wiring):
    request_cls = MagicMock()
    wiring = ("TEXT", "IMAGES", request_cls)
    get_wiring = mocker.patch.object(_dw_module, "get_wiring", return_value=wiring)
    mocker.patch.object(_dw_module, "get_register_model_kwargs", return_value={"worker_type": "AGGREGATED"})
    register_model = AsyncMock()
    mocker.patch.dict("sys.modules", {"dynamo.llm": MagicMock(register_model=register_model)})
    config = _dw_module.DynamoWorkerConfig(type="image", model_path="org/model", model_name="served-model")
    worker = _dw_module._HighLevelDynamoWorker(MagicMock(), config)
    if cached_wiring:
        worker._wiring = wiring

    endpoint = MagicMock()
    asyncio.run(worker.on_ready(MagicMock(), endpoint))

    register_model.assert_awaited_once_with(
        "TEXT",
        "IMAGES",
        endpoint,
        "org/model",
        model_name="served-model",
        worker_type="AGGREGATED",
    )
    assert get_wiring.called is not cached_wiring


def test_high_level_callable_no_mapping_passes_raw_request(mocker):
    """Callable without mapping receives raw request object."""
    received = []

    def my_fn(request):
        received.append(request)
        return {"object": "list", "data": [], "model": "m", "usage": {}}

    mock_request = {"input": "hello", "model": "org/model"}

    mock_run = mocker.patch.object(_dw_module, "run_dynamo_worker")
    typed_request = MagicMock(**mock_request)
    mock_wiring_req = MagicMock(return_value=typed_request)
    mocker.patch.object(_dw_module, "get_wiring", return_value=(MagicMock(), MagicMock(), mock_wiring_req))

    from aitune.dynamo.worker import DynamoWorkerConfig, dynamo_worker

    cfg = DynamoWorkerConfig(type="embedding", model_path="org/model")
    dynamo_worker(my_fn, cfg)

    # Extract the serve function that was passed to run_dynamo_worker
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
    mock_wiring_req.assert_called_once_with(**mock_request)
    assert received[0] is typed_request


@pytest.mark.parametrize("use_mapping", [False, True])
def test_high_level_awaits_async_callable_result(mocker, use_mapping):
    """Async callables are awaited after invocation through the executor."""
    typed_request = MagicMock()
    request_cls = MagicMock(return_value=typed_request)
    mocker.patch.object(_dw_module, "get_wiring", return_value=(MagicMock(), MagicMock(), request_cls))
    result = {"object": "list", "data": [], "model": "m", "usage": {}}
    model = AsyncMock(return_value=result)
    mapping = MagicMock(return_value={"request": typed_request}) if use_mapping else None
    config = _dw_module.DynamoWorkerConfig(type="embedding", model_path="org/model", mapping=mapping)
    worker = _dw_module._HighLevelDynamoWorker(model, config)
    request = {"input": "hello", "model": "org/model"}

    async def collect():
        return [chunk async for chunk in worker.serve(request)]

    assert asyncio.run(collect()) == [result]
    request_cls.assert_called_once_with(**request)
    if use_mapping:
        assert mapping is not None
        mapping.assert_called_once_with(typed_request)
        model.assert_awaited_once_with(request=typed_request)
    else:
        model.assert_awaited_once_with(typed_request)


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

    mocker.patch.object(_dw_module, "get_wiring", return_value=(MagicMock(), MagicMock(), mock_request_cls))
    mock_run = mocker.patch.object(_dw_module, "run_dynamo_worker")

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
    mocker.patch.object(_dw_module, "get_wiring", return_value=(MagicMock(), MagicMock(), mock_request_cls))
    mock_run = mocker.patch.object(_dw_module, "run_dynamo_worker")

    from aitune.dynamo.worker import DynamoWorkerConfig, dynamo_worker

    cfg = DynamoWorkerConfig(type="image", model_path="org/model", mapping=bad_mapping)
    dynamo_worker(my_fn, cfg)

    serve_fn = mock_run.call_args.kwargs.get("serve") or mock_run.call_args[0][1]

    async def collect():
        async for _ in serve_fn({}):
            pass

    with pytest.raises(TypeError):
        asyncio.run(collect())


def test_high_level_collective_follower_executes_without_packing_response(mocker):
    """Collective followers execute the callable but leave response packing to rank zero."""
    called = []

    def my_fn(request):
        called.append(request)
        return None

    request = MagicMock()
    request_cls = MagicMock(return_value=request)
    mocker.patch.object(_dw_module, "get_wiring", return_value=(MagicMock(), MagicMock(), request_cls))
    mocker.patch.object(_dw_module, "is_rank_zero", return_value=False)

    config = _dw_module.DynamoWorkerConfig(type="image", model_path="org/model")
    worker = _dw_module._HighLevelDynamoWorker(my_fn, config)
    worker._collective = True

    async def collect():
        return [chunk async for chunk in worker.serve({"prompt": "sunset"})]

    assert asyncio.run(collect()) == []
    assert called == [request]


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_exports_accessible_from_aitune_dynamo():
    """dynamo_worker, DynamoWorker, DynamoWorkerConfig are exported from aitune.dynamo."""
    import aitune.dynamo as aid

    assert hasattr(aid, "dynamo_worker")
    assert hasattr(aid, "DynamoWorker")
    assert hasattr(aid, "DynamoWorkerConfig")
