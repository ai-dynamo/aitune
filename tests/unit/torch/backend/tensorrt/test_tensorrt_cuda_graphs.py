# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TensorRT CUDA Graphs implementation."""

import weakref
from unittest.mock import Mock

import pytest
import torch

from aitune.torch.backend.tensorrt.cuda_graphs import CudaGraphProfile, TensorRTCudaGraphCache
from aitune.torch.backend.tensorrt.tensorrt_backend import TensorRTBackend, TensorRTBackendConfig
from tests.utilities.helpers import requires_cuda


@requires_cuda
def test_sanity_check_cuda_graphs():
    """Simple cuda graph capture and replay."""
    device = torch.device("cuda")

    class AddOneModule(torch.nn.Module):
        def forward(self, x):
            return x + 1

    model = AddOneModule()
    model.to(torch.device("cuda"))
    model.eval()
    data = torch.ones((3, 1), device=device)  # Device and dtype must be the sames

    stream = torch.cuda.Stream()

    # Warmup
    stream.synchronize()
    with torch.cuda.stream(stream):
        output = model(data)
    stream.synchronize()

    torch.testing.assert_close(output, torch.tensor([[2.0], [2.0], [2.0]], device=device))

    # CUDA Graphs capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = model(data)

    # runs
    graph.replay()
    torch.testing.assert_close(output, torch.tensor([[2.0], [2.0], [2.0]], device=device))

    data.copy_(torch.tensor([[3.0], [3.0], [3.0]]))
    graph.replay()
    torch.testing.assert_close(output, torch.tensor([[4.0], [4.0], [4.0]], device=device))


@pytest.mark.parametrize("policy", ["lfu", "lru"])
def test_config_serialization(policy):
    """Test configuration serialization with CUDA graphs."""
    config = TensorRTBackendConfig(use_cuda_graphs=True, max_cuda_graphs=2, cuda_graph_cache_policy=policy)
    config_dict = config.to_dict()

    assert "use_cuda_graphs" in config_dict
    assert config_dict["use_cuda_graphs"] is True
    assert config_dict["max_cuda_graphs"] == 2
    assert TensorRTBackendConfig.from_dict(config_dict).cuda_graph_cache_policy == policy
    assert TensorRTBackendConfig.from_dict(config_dict).max_cuda_graphs == 2


def test_config_deserialization():
    """Test configuration deserialization with CUDA graphs."""
    config_dict = {"use_cuda_graphs": True}
    config = TensorRTBackendConfig.from_dict(config_dict)

    assert config.use_cuda_graphs is True


@pytest.fixture
def trtre_backend():
    """Set up test fixtures."""
    config = TensorRTBackendConfig(use_cuda_graphs=True)
    backend = TensorRTBackend(config=config)

    return backend


def test_deactivate_cleanup_cuda_graphs(trtre_backend):
    """Test that _deactivate properly cleans up CUDA graph variables."""
    # Set some dummy values
    entry = CudaGraphProfile(Mock(), Mock(), graph=Mock())
    trtre_backend._cuda_graphs.profiles[0] = entry
    trtre_backend._cuda_graphs.active = entry

    # Call deactivate
    trtre_backend._deactivate()

    # Check that CUDA graph variables are cleaned up
    assert trtre_backend._cuda_graphs.active is None
    assert not trtre_backend._cuda_graphs.profiles
    assert not trtre_backend._cuda_graphs.static_profile_indices


@pytest.fixture
def trtre_backend_mock():
    """Set up test fixtures."""
    config = TensorRTBackendConfig(use_cuda_graphs=True)
    backend = TensorRTBackend(config=config)

    backend._context = Mock()
    backend._cuda_stream = Mock()
    backend._cuda_graphs.static_profile_indices = {0}
    profile = CudaGraphProfile(backend._context, Mock())
    backend._cuda_graphs.profiles[0] = profile
    backend._cuda_graphs.active = profile

    return backend


def test_execute_with_cuda_graphs_first_run(trtre_backend_mock: TensorRTBackend, mocker):
    """Test CUDA graph execution on first run (capture phase)."""
    # Mock successful execution
    mock_graph = Mock()
    mocker.patch("torch.cuda.CUDAGraph", return_value=mock_graph)
    capture = mocker.patch("torch.cuda.graph")
    trtre_backend_mock._context.execute_async_v3.return_value = True

    # Execute
    trtre_backend_mock._cuda_graphs.execute(trtre_backend_mock._cuda_stream)

    # Verify capture sequence
    assert trtre_backend_mock._context.execute_async_v3.call_count == 2  # Setup + capture
    capture.assert_called_once_with(mock_graph, stream=trtre_backend_mock._cuda_stream)
    capture.return_value.__exit__.assert_called_once()
    mock_graph.replay.assert_called_once()

    assert trtre_backend_mock._cuda_graphs.profiles[0].graph is mock_graph
    trtre_backend_mock._cuda_graphs.execute(trtre_backend_mock._cuda_stream)
    assert mock_graph.replay.call_count == 2
    assert trtre_backend_mock._context.execute_async_v3.call_count == 2


def test_execute_with_cuda_graphs_subsequent_run(trtre_backend_mock):
    """Test CUDA graph execution on subsequent runs (launch phase)."""
    # Set up existing CUDA graph
    trtre_backend_mock._cuda_graphs.active.graph = Mock()

    # Execute
    trtre_backend_mock._cuda_graphs.execute(trtre_backend_mock._cuda_stream)

    # Verify launch
    trtre_backend_mock._cuda_graphs.active.graph.replay.assert_called_once()


def test_execute_with_cuda_graphs_capture_failure(trtre_backend_mock, mocker):
    """Test CUDA graph execution handles capture failure."""

    mock_graph = Mock()
    mocker.patch("torch.cuda.CUDAGraph", return_value=mock_graph)
    capture = mocker.patch("torch.cuda.graph")

    trtre_backend_mock._context.execute_async_v3.side_effect = [True, False, True]
    config_key = trtre_backend_mock.key()

    trtre_backend_mock._cuda_graphs.execute(trtre_backend_mock._cuda_stream)

    assert trtre_backend_mock._context.execute_async_v3.call_count == 3
    capture.return_value.__exit__.assert_called_once()
    mock_graph.replay.assert_not_called()
    assert not trtre_backend_mock._use_cuda_graphs
    assert trtre_backend_mock._cuda_graphs.active.graph is None
    assert trtre_backend_mock.key() == config_key
    # A new input shape must not trigger another capture after failure.
    trtre_backend_mock._set_optimization_profiles({"input1": torch.empty(2, 3)})
    assert not trtre_backend_mock._use_cuda_graphs


def test_describe_with_cuda_graphs():
    """Test an explicit opt-out is visible in the backend description."""
    config = TensorRTBackendConfig(use_cuda_graphs=False)
    backend = TensorRTBackend(config=config)

    description = backend.describe()
    assert "use_cuda_graphs=False" in description or "use_cuda_graphs: False" in description


def test_cuda_graphs_enabled_by_default():
    assert TensorRTBackendConfig().use_cuda_graphs is True
    assert TensorRTBackendConfig.from_dict({}).max_cuda_graphs == 8
    assert TensorRTBackendConfig.from_dict({}).cuda_graph_cache_policy == "lfu"
    assert TensorRTCudaGraphCache().policy == "lfu"
    # Eligibility is known only after activation loads the engine and profiles.
    assert not TensorRTBackend()._use_cuda_graphs
    assert not TensorRTBackend(TensorRTBackendConfig(use_cuda_graphs=False))._use_cuda_graphs


@pytest.mark.parametrize("failure_stage", ["create", "begin", "end"])
def test_capture_exception_falls_back(trtre_backend_mock, mocker, failure_stage):
    create = mocker.patch("torch.cuda.CUDAGraph")
    capture = mocker.patch("torch.cuda.graph")
    failing_call = {"create": create, "begin": capture.return_value.__enter__, "end": capture.return_value.__exit__}
    failing_call[failure_stage].side_effect = RuntimeError("capture unavailable")
    trtre_backend_mock._context.execute_async_v3.return_value = True

    trtre_backend_mock._cuda_graphs.execute(trtre_backend_mock._cuda_stream)

    assert not trtre_backend_mock._use_cuda_graphs
    assert trtre_backend_mock._context.execute_async_v3.call_count == (3 if failure_stage == "end" else 2)


def test_normal_execution_failure_after_capture_failure_propagates(trtre_backend_mock, mocker):
    mocker.patch("torch.cuda.CUDAGraph")
    mocker.patch("torch.cuda.graph")
    trtre_backend_mock._context.execute_async_v3.side_effect = [True, False, False]

    with pytest.raises(RuntimeError, match="TensorRT execution failed$"):
        trtre_backend_mock._cuda_graphs.execute(trtre_backend_mock._cuda_stream)


def test_setup_failure_does_not_attempt_capture(trtre_backend_mock, mocker):
    capture = mocker.patch("torch.cuda.CUDAGraph")
    trtre_backend_mock._context.execute_async_v3.return_value = False

    with pytest.raises(RuntimeError, match="TensorRT execution failed during CUDA graph setup"):
        trtre_backend_mock._cuda_graphs.execute(trtre_backend_mock._cuda_stream)

    capture.assert_not_called()


def test_replay_failure_propagates(trtre_backend_mock, mocker):
    graph = mocker.patch("torch.cuda.CUDAGraph").return_value
    mocker.patch("torch.cuda.graph")
    graph.replay.side_effect = RuntimeError("replay failed")

    with pytest.raises(RuntimeError, match="replay failed"):
        trtre_backend_mock._cuda_graphs.execute(trtre_backend_mock._cuda_stream)


def test_inference_continues_without_recapturing_after_failure(trtre_backend_mock, mocker):
    backend = trtre_backend_mock
    graph = mocker.patch("torch.cuda.CUDAGraph")
    mocker.patch("torch.cuda.graph")
    mocker.patch("torch.cuda.stream")
    backend._start_time = Mock()
    backend._end_time = Mock()
    backend._start_time.elapsed_time.return_value = 1.0
    mocker.patch.object(backend, "_set_optimization_profiles")
    mocker.patch.object(backend, "_set_input_tensors")
    mocker.patch.object(backend, "_prepare_outputs_for_return")
    mocker.patch.object(
        backend,
        "_prepare_inputs",
        side_effect=[
            {"input": torch.empty(1, 3)},
            {"input": torch.empty(1, 3)},
            {"input": torch.empty(2, 3)},
        ],
    )
    backend._context.execute_async_v3.side_effect = [True, False, True, True, True]

    for _ in range(3):
        backend._infer()

    graph.assert_called_once()
    assert backend._context.execute_async_v3.call_count == 5
    assert backend._prepare_outputs_for_return.call_count == 3


@pytest.mark.parametrize("capture_fails", [False, True])
def test_static_inputs_live_until_synchronization(trtre_backend_mock, mocker, capture_fails):
    backend = trtre_backend_mock
    mocker.patch("torch.cuda.CUDAGraph")
    mocker.patch("torch.cuda.graph")
    mocker.patch("torch.cuda.stream")
    backend._start_time = Mock()
    backend._end_time = Mock()
    backend._start_time.elapsed_time.return_value = 1.0
    mocker.patch.object(backend, "_prepare_inputs", return_value={"input": torch.empty(1, 3)})
    mocker.patch.object(backend, "_set_optimization_profiles")
    mocker.patch.object(backend, "_prepare_outputs_for_return")
    buffers = []

    def set_inputs(inputs):
        tensor = torch.empty_like(inputs["input"])
        backend._cuda_graphs.active.inputs["input"] = tensor
        buffers.append(weakref.ref(tensor))

    def synchronize():
        # The backend must own the buffers until queued inference has finished consuming them.
        assert buffers[0]() is not None

    mocker.patch.object(backend, "_set_input_tensors", side_effect=set_inputs)
    backend._cuda_stream.synchronize.side_effect = synchronize
    backend._context.execute_async_v3.side_effect = [True, False, True] if capture_fails else [True, True]

    backend._infer()

    backend._cuda_stream.synchronize.assert_called_once()
    assert (buffers[0]() is None) is capture_fails
    assert (backend._cuda_graphs.active is not None) is not capture_fails


@pytest.mark.parametrize(
    ("profiles", "engine_shape", "shape_input", "expected"),
    [
        ([], (2, 3), False, {0}),
        ([], (-1, 3), False, set()),
        ([{"x": ((2, 3),) * 3}], (-1, 3), False, {0}),
        ([{"x": ((1, 3), (2, 3), (4, 3))}], (-1, 3), False, set()),
        ([{"x": ((2, 3), (4, 3), (2, 3))}], (-1, 3), False, set()),
        ([{"x": ((-1, 3),) * 3}], (-1, 3), False, set()),
        ([{"x": ((2, 3),) * 3}], (-1, 3), True, set()),
        ([{"other": ((2, 3),) * 3}], (-1, 3), False, set()),
        ([{"x": ((2, 3),) * 3}, {"x": ((1, 3), (2, 3), (4, 3))}], (-1, 3), False, {0}),
    ],
)
def test_static_profile_eligibility(trtre_backend, profiles, engine_shape, shape_input, expected):
    backend = trtre_backend
    backend._base_context = Mock()
    backend._base_context.engine.is_shape_inference_io.return_value = shape_input
    backend._input_names = ["x"]
    backend._io_tensors = {"x": {"shape": engine_shape}}
    backend._trt_optimization_profiles = profiles
    backend._cuda_graphs.configure(backend._base_context.engine, backend._io_tensors, backend._input_names, profiles)
    assert backend._cuda_graphs.static_profile_indices == expected


def test_all_inputs_must_be_static(trtre_backend):
    backend = trtre_backend
    backend._base_context = Mock()
    backend._base_context.engine.is_shape_inference_io.return_value = False
    backend._input_names = ["x", "y"]
    backend._trt_optimization_profiles = [
        {"x": ((2, 3),) * 3, "y": ((1,), (2,), (4,))},
        {"x": ((2, 3),) * 3, "y": ((2,),) * 3},
    ]
    backend._cuda_graphs.configure(
        backend._base_context.engine, {}, backend._input_names, backend._trt_optimization_profiles
    )
    assert backend._cuda_graphs.static_profile_indices == {1}


@pytest.fixture
def profile_backend(trtre_backend, mocker):
    backend = trtre_backend
    backend._base_context = Mock(active_optimization_profile=0)
    backend._base_output_allocator = Mock()
    backend._cuda_stream = Mock()
    backend._cuda_graphs.static_profile_indices = {0, 1}
    backend._trt_optimization_profiles = [
        {"x": ((2, 3),) * 3},
        {"x": ((4, 3),) * 3},
        {"x": ((5, 3), (6, 3), (8, 3))},
    ]
    backend._base_context.engine.create_execution_context.side_effect = lambda: Mock()
    mocker.patch.object(backend, "_create_output_allocator", side_effect=lambda context: Mock())
    return backend


def test_profile_cache_survives_static_and_dynamic_switches(profile_backend):
    backend = profile_backend
    entries = {}
    for index, size in [(0, 2), (1, 4), (2, 6), (0, 2), (1, 4)]:
        backend._set_optimization_profiles({"x": torch.empty(size, 3)})
        if index == 2:
            assert not backend._use_cuda_graphs
            assert backend._context is backend._base_context
            assert backend._output_allocator is backend._base_output_allocator
            assert backend._cuda_graphs.active is None
            continue
        assert backend._use_cuda_graphs
        entry = backend._cuda_graphs.profiles[index]
        if index not in entries:
            assert backend._cuda_graphs.active.graph is None
            entry.graph = Mock()
            entry.inputs["x"] = torch.empty(size, 3)
            entries[index] = entry
        else:
            assert entry is entries[index]
            assert backend._cuda_graphs.active.graph is entry.graph
        assert backend._context is entry.context
        assert backend._output_allocator is entry.output_allocator
        assert backend._cuda_graphs.active.inputs is entry.inputs
        entry.context.set_optimization_profile_async.assert_called_once_with(index, backend._cuda_stream.cuda_stream)
    assert entries[0].context is not entries[1].context
    assert entries[0].output_allocator is not entries[1].output_allocator
    assert backend._base_context.engine.create_execution_context.call_count == 2
    backend._base_context.set_optimization_profile_async.assert_called_once_with(2, backend._cuda_stream.cuda_stream)


@pytest.mark.parametrize("disabled_by", ["config", "capture_failure"])
def test_disabled_graphs_use_base_context(profile_backend, disabled_by):
    backend = profile_backend
    if disabled_by == "config":
        backend._config.use_cuda_graphs = False
    else:
        backend._cuda_graphs.capture_failed = True
    backend._set_optimization_profiles({"x": torch.empty(4, 3)})
    assert backend._context is backend._base_context
    assert not backend._use_cuda_graphs
    assert not backend._cuda_graphs.profiles
    backend._base_context.engine.create_execution_context.assert_not_called()


def test_replay_does_not_rebind_context(trtre_backend_mock, mocker):
    backend = trtre_backend_mock
    backend._cuda_graphs.active.graph = Mock()
    backend._input_names = ["x"]
    backend._io_tensors = {"x": {"dtype": torch.float32}}
    copy = mocker.patch.object(backend._cuda_graphs, "copy_input", side_effect=lambda name, value, stream: value)
    value = Mock(shape=(2, 3), is_cuda=True)
    value.is_contiguous.return_value = True
    backend._set_input_tensors({"x": value})
    copy.assert_called_once()
    backend._context.set_input_shape.assert_not_called()
    backend._context.set_tensor_address.assert_not_called()


def test_input_copy_reuses_profile_buffer_on_inference_stream(trtre_backend_mock, mocker):
    backend = trtre_backend_mock
    current_stream = mocker.patch("torch.cuda.current_stream").return_value
    stream = mocker.patch("torch.cuda.stream")
    value = torch.empty(2, 3)
    first = backend._cuda_graphs.copy_input("x", value, backend._cuda_stream)
    assert backend._cuda_graphs.copy_input("x", value, backend._cuda_stream) is first
    backend._cuda_stream.wait_stream.assert_called_with(current_stream)
    stream.assert_called_with(backend._cuda_stream)
    with pytest.raises(RuntimeError, match="captured profile shape"):
        backend._cuda_graphs.copy_input("x", torch.empty(1, 3), backend._cuda_stream)


@pytest.mark.parametrize("shape", [(4, 3), (2, 3, 1)])
def test_single_static_profile_rejects_other_shapes(profile_backend, shape):
    backend = profile_backend
    backend._trt_optimization_profiles = backend._trt_optimization_profiles[:1]
    with pytest.raises(RuntimeError, match="No TensorRT optimization profile"):
        backend._set_optimization_profiles({"x": torch.empty(shape)})
    backend._base_context.engine.create_execution_context.assert_not_called()


def test_deactivate_releases_all_profile_resources(profile_backend):
    backend = profile_backend
    for size in (2, 4):
        backend._set_optimization_profiles({"x": torch.empty(size, 3)})
        entry = backend._cuda_graphs.active
        entry.graph = Mock()
        entry.inputs["x"] = torch.empty(size, 3)
    entries = list(backend._cuda_graphs.profiles.values())
    backend._context = None  # Native context manager teardown is outside this unit test.
    stream = backend._cuda_stream
    stream.synchronize.side_effect = lambda: all(entry.inputs for entry in entries) or pytest.fail("Released too early")
    backend._deactivate()
    stream.synchronize.assert_called_once()
    assert not backend._cuda_graphs.profiles
    assert all(entry.graph is None and not entry.inputs for entry in entries)
    assert backend._base_context is None


def test_capture_failure_releases_cached_profiles_after_execution(profile_backend, mocker):
    backend = profile_backend
    mocker.patch("torch.cuda.CUDAGraph")
    mocker.patch("torch.cuda.graph")
    mocker.patch("torch.cuda.stream")
    backend._start_time = Mock()
    backend._end_time = Mock()
    backend._start_time.elapsed_time.return_value = 1.0
    backend._set_optimization_profiles({"x": torch.empty(2, 3)})
    first = backend._cuda_graphs.profiles[0]
    first.graph = Mock()
    first.inputs["x"] = torch.empty(2, 3)
    backend._set_optimization_profiles({"x": torch.empty(4, 3)})
    second = backend._cuda_graphs.profiles[1]
    second.inputs["x"] = torch.empty(4, 3)
    second.context.execute_async_v3.side_effect = [True, False, True]
    mocker.patch.object(backend, "_prepare_inputs", return_value={"x": torch.empty(4, 3)})
    mocker.patch.object(backend, "_set_input_tensors")
    mocker.patch.object(backend, "_prepare_outputs_for_return")

    def check_lifetime():
        assert first.graph is not None
        assert first.inputs and second.inputs

    backend._cuda_stream.synchronize.side_effect = check_lifetime
    backend._infer()
    assert backend._cuda_graphs.capture_failed
    assert not backend._cuda_graphs.profiles
    assert first.graph is None and not first.inputs and not second.inputs
    second.context.execute_async_v3.assert_called()
    backend._set_optimization_profiles({"x": torch.empty(2, 3)})
    assert backend._context is backend._base_context
    assert not backend._use_cuda_graphs
    assert backend._base_context.engine.create_execution_context.call_count == 2


@pytest.mark.parametrize("failure", ["context", "profile"])
def test_profile_context_setup_failure_propagates(profile_backend, failure):
    backend = profile_backend
    create = backend._base_context.engine.create_execution_context
    create.side_effect = None
    create.return_value = None if failure == "context" else Mock()
    if failure == "profile":
        create.return_value.set_optimization_profile_async.return_value = False
    with pytest.raises(RuntimeError, match="execution context|rejected optimization profile"):
        backend._set_optimization_profiles({"x": torch.empty(2, 3)})
    assert not backend._cuda_graphs.profiles
    assert not backend._cuda_graphs.capture_failed


def test_activation_starts_with_empty_profile_cache(trtre_backend, mocker):
    backend = trtre_backend
    backend._config.max_cuda_graphs = 2
    backend._config.cuda_graph_cache_policy = "lru"
    backend._engine_artifact = Mock()
    backend._trt_optimization_profiles_artifact = Mock()
    context = mocker.MagicMock()
    context.engine.is_shape_inference_io.return_value = False
    runtime = mocker.patch("aitune.torch.backend.tensorrt.tensorrt_backend.TensorRTRuntime").return_value
    runtime.create_execution_context.return_value = (context, {"x": {"shape": (2, 3)}}, ["x"], ["y"], Mock())
    mocker.patch("aitune.torch.backend.tensorrt.tensorrt_backend.cuda_set_device")
    mocker.patch("torch.cuda.Stream")
    mocker.patch("torch.cuda.Event")
    mocker.patch.object(backend, "_create_output_allocator", side_effect=lambda context: Mock())
    mocker.patch.object(backend, "_load_trt_optimization_profiles", return_value=[{"x": ((2, 3),) * 3}])
    backend._activate()
    assert backend._cuda_graphs.max_graphs == 2
    assert backend._cuda_graphs.policy == "lru"
    assert backend._cuda_graphs.static_profile_indices == {0}
    assert not backend._cuda_graphs.profiles
    backend._set_optimization_profiles({"x": torch.empty(2, 3)})
    entry = backend._cuda_graphs.profiles[0]
    entry.graph = Mock()
    backend._deactivate()
    backend._activate()
    assert backend._cuda_graphs.max_graphs == 2
    assert backend._cuda_graphs.policy == "lru"
    assert backend._cuda_graphs.static_profile_indices == {0}
    assert not backend._cuda_graphs.profiles
    assert backend._cuda_graphs.active is None
    backend._set_optimization_profiles({"x": torch.empty(2, 3)})
    assert backend._cuda_graphs.profiles[0] is not entry
    assert backend._cuda_graphs.profiles[0].graph is None


def test_graph_manager_preserves_capture_failure_when_reconfigured():
    graphs = TensorRTCudaGraphCache()
    engine = Mock()
    engine.is_shape_inference_io.return_value = False
    profiles = [{"x": ((2, 3),) * 3}]
    graphs.configure(engine, {}, ["x"], profiles)
    assert graphs.is_eligible(0)
    graphs.capture_failed = True
    graphs.clear()
    graphs.configure(engine, {}, ["x"], profiles)
    assert not graphs.is_eligible(0)
    assert graphs.capture_failed
    assert graphs.active is None
    assert not graphs.profiles


@pytest.mark.parametrize("capacity", [0, -1, 1.5, True])
def test_invalid_graph_cache_capacity(capacity):
    with pytest.raises(ValueError, match="max_cuda_graphs must be a positive integer"):
        TensorRTBackendConfig.from_dict({"max_cuda_graphs": capacity})


def test_lru_cache_retains_recently_used_profiles_and_recaptures_evicted(profile_backend, mocker):
    backend = profile_backend
    graphs = backend._cuda_graphs
    graphs.policy = "lru"
    graphs.static_profile_indices = set(range(9))
    backend._trt_optimization_profiles = [{"x": ((size, 3),) * 3} for size in range(1, 10)]
    capture = mocker.patch("torch.cuda.CUDAGraph", side_effect=lambda: Mock())
    mocker.patch("torch.cuda.graph")

    def infer_profile(size):
        backend._set_optimization_profiles({"x": torch.empty(size, 3)})
        graphs.execute(backend._cuda_stream)
        assert len(graphs.profiles) <= 8

    for size in range(1, 9):
        infer_profile(size)
    first = graphs.profiles[0]
    evicted = graphs.profiles[1]
    infer_profile(1)  # A cache hit must update recency; profile 1 is now the oldest.
    assert capture.call_count == 8
    infer_profile(9)
    assert list(graphs.profiles) == [2, 3, 4, 5, 6, 7, 0, 8]
    assert graphs.profiles[0] is first
    assert evicted.graph is None
    evicted.output_allocator.clear.assert_called_once()
    infer_profile(2)
    assert graphs.profiles[1] is not evicted
    assert capture.call_count == 10
    assert list(graphs.profiles) == [3, 4, 5, 6, 7, 0, 8, 1]
    assert len(backend._trt_optimization_profiles) == 9
    assert not graphs.capture_failed


def test_eviction_synchronizes_and_releases_resources_before_allocating(profile_backend):
    backend = profile_backend
    graphs = backend._cuda_graphs
    graphs.policy = "lru"
    graphs.max_graphs = 1
    backend._set_optimization_profiles({"x": torch.empty(2, 3)})
    old = graphs.active
    old.graph = Mock()
    old.inputs["x"] = torch.empty(2, 3)

    def synchronize():
        assert old.graph is not None and old.inputs
        old.output_allocator.clear.assert_not_called()

    def create_context():
        assert old.graph is None and not old.inputs
        old.output_allocator.clear.assert_called_once()
        assert backend._context is backend._base_context
        assert backend._output_allocator is backend._base_output_allocator
        assert graphs.active is None and not graphs.profiles
        return Mock()

    backend._cuda_stream.synchronize.side_effect = synchronize
    backend._base_context.engine.create_execution_context.side_effect = create_context
    backend._set_optimization_profiles({"x": torch.empty(4, 3)})
    backend._cuda_stream.synchronize.assert_called_once()
    assert list(graphs.profiles) == [1]


def test_eviction_preserves_resources_if_synchronization_fails(profile_backend):
    backend = profile_backend
    graphs = backend._cuda_graphs
    graphs.policy = "lru"
    graphs.max_graphs = 1
    backend._set_optimization_profiles({"x": torch.empty(2, 3)})
    old = graphs.active
    old.graph = Mock()
    old.inputs["x"] = torch.empty(2, 3)
    backend._cuda_stream.synchronize.side_effect = RuntimeError("synchronization failed")
    with pytest.raises(RuntimeError, match="synchronization failed"):
        backend._set_optimization_profiles({"x": torch.empty(4, 3)})
    assert graphs.profiles[0] is old
    assert old.graph is not None and old.inputs
    old.output_allocator.clear.assert_not_called()
    assert backend._base_context.engine.create_execution_context.call_count == 1


@pytest.mark.parametrize("policy", ["fifo", "LFU", "", None])
def test_invalid_graph_cache_policy(policy):
    with pytest.raises(ValueError, match="cuda_graph_cache_policy"):
        TensorRTBackendConfig.from_dict({"cuda_graph_cache_policy": policy})


@pytest.fixture
def lfu_backend(profile_backend):
    backend = profile_backend
    assert backend._cuda_graphs.policy == "lfu"
    backend._cuda_graphs.max_graphs = 2
    backend._cuda_graphs.static_profile_indices = set(range(10))
    backend._trt_optimization_profiles = [{"x": ((size, 3),) * 3} for size in range(1, 11)]
    return backend


def test_lfu_protects_hot_profiles_from_scan_and_admits_repeated_candidate(lfu_backend):
    backend = lfu_backend
    graphs = backend._cuda_graphs

    def request(index):
        backend._set_optimization_profiles({"x": torch.empty(index + 1, 3)})
        return graphs.active

    hot = request(0)
    for _ in range(4):
        assert request(0) is hot
    cold = request(1)  # More recent than the hot profile, but less frequent.
    for index in range(2, 10):
        assert request(index) is None
        assert backend._context is backend._base_context
        assert backend._output_allocator is backend._base_output_allocator
        assert not backend._use_cuda_graphs
        backend._base_context.set_optimization_profile_async.assert_called_with(index, backend._cuda_stream.cuda_stream)
    assert graphs.profiles == {0: hot, 1: cold}
    assert backend._base_context.engine.create_execution_context.call_count == 2
    backend._cuda_stream.synchronize.assert_not_called()
    assert not graphs.capture_failed
    assert request(2) is not None  # The rejected request contributed to frequency.
    assert list(graphs.profiles) == [0, 2]
    cold.output_allocator.clear.assert_called_once()
    assert request(0) is hot


def test_lfu_breaks_victim_frequency_ties_by_recency(lfu_backend):
    backend = lfu_backend
    graphs = backend._cuda_graphs
    for index in (0, 1, 1, 0):
        backend._set_optimization_profiles({"x": torch.empty(index + 1, 3)})
    assert list(graphs.profiles) == [1, 0]
    for _ in range(2):
        backend._set_optimization_profiles({"x": torch.empty(3, 3)})
        assert graphs.active is None  # Equal frequency is insufficient for admission.
    backend._set_optimization_profiles({"x": torch.empty(3, 3)})
    assert list(graphs.profiles) == [0, 2]


def test_lfu_aging_allows_new_hot_profile_to_replace_historical_hot_profile(lfu_backend):
    backend = lfu_backend
    graphs = backend._cuda_graphs
    graphs.max_graphs = 1

    def request(index):
        return graphs.select(
            index, backend._base_context.engine, backend._cuda_stream, backend._create_output_allocator
        )

    for _ in range(1023):
        assert request(0) is not None
    historical = graphs.profiles[0]
    for _ in range(511):
        assert request(1) is None
    assert request(1) is not None  # Decay reduced historical popularity from 1023 to 511.
    assert list(graphs.profiles) == [1]
    historical.output_allocator.clear.assert_called_once()
    assert graphs._frequencies == {0: 511, 1: 512}
    graphs.clear()
    assert not graphs._frequencies and graphs._requests_since_decay == 0


def test_lfu_rejected_request_executes_normally_and_cached_graph_still_replays(lfu_backend, mocker):
    backend = lfu_backend
    graphs = backend._cuda_graphs
    graphs.max_graphs = 1
    backend._context = backend._base_context
    backend._start_time = Mock()
    backend._end_time = Mock()
    mocker.patch("torch.cuda.CUDAGraph", side_effect=lambda: Mock())
    mocker.patch("torch.cuda.graph")
    mocker.patch("torch.cuda.stream")
    mocker.patch.object(backend, "_set_input_tensors")
    mocker.patch.object(backend, "_prepare_outputs_for_return")
    inputs = mocker.patch.object(backend, "_prepare_inputs", return_value={"x": torch.empty(1, 3)})
    backend._infer()
    entry = graphs.profiles[0]
    assert entry.graph.replay.call_count == 1
    inputs.return_value = {"x": torch.empty(2, 3)}
    backend._infer()
    assert graphs.active is None
    backend._base_context.execute_async_v3.assert_called_once_with(backend._cuda_stream.cuda_stream)
    assert not graphs.capture_failed
    inputs.return_value = {"x": torch.empty(1, 3)}
    backend._infer()
    assert graphs.active is entry
    assert entry.graph.replay.call_count == 2
