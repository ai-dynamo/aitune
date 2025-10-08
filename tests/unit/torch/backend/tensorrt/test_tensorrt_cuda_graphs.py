# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for TensorRT CUDA Graphs implementation."""

from unittest.mock import Mock

import pytest
import torch

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


@requires_cuda
def test_config_serialization():
    """Test configuration serialization with CUDA graphs."""
    config = TensorRTBackendConfig(use_cuda_graphs=True)
    config_dict = config.to_dict()

    assert "use_cuda_graphs" in config_dict
    assert config_dict["use_cuda_graphs"] is True


@requires_cuda
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


@requires_cuda
def test_invalidate_cuda_graph(trtre_backend):
    """Test CUDA graph invalidation."""
    # Set some dummy values
    trtre_backend._cuda_graph = Mock()
    trtre_backend._last_input_shapes = {"input1": torch.Size([2, 3, 224, 224])}

    # Invalidate the graph
    trtre_backend._invalidate_cuda_graph({"input1": torch.randn(1, 3, 224, 224)})

    # Check that graph is invalidated
    assert trtre_backend._cuda_graph is None


@requires_cuda
def test_deactivate_cleanup_cuda_graphs(trtre_backend):
    """Test that _deactivate properly cleans up CUDA graph variables."""
    # Set some dummy values
    trtre_backend._cuda_graph = Mock()
    trtre_backend._last_input_shapes = {"input1": torch.Size([1, 3, 224, 224])}

    # Call deactivate
    trtre_backend._deactivate()

    # Check that CUDA graph variables are cleaned up
    assert not hasattr(trtre_backend, "_cuda_graph")
    assert not hasattr(trtre_backend, "_last_input_shapes")


@pytest.fixture
def trtre_backend_mock():
    """Set up test fixtures."""
    config = TensorRTBackendConfig(use_cuda_graphs=True)
    backend = TensorRTBackend(config=config)

    backend._context = Mock()
    backend._cuda_stream = torch.cuda.Stream()

    return backend


@requires_cuda
def test_execute_with_cuda_graphs_first_run(trtre_backend_mock: TensorRTBackend, mocker):
    """Test CUDA graph execution on first run (capture phase)."""
    # Mock successful execution
    mock_graph = Mock()
    mocker.patch("torch.cuda.CUDAGraph", return_value=mock_graph)
    trtre_backend_mock._context.execute_async_v3.return_value = True
    trtre_backend_mock._infer_cuda_graph = trtre_backend_mock._build_cuda_graph

    # Execute
    trtre_backend_mock._infer_cuda_graph()

    # Verify capture sequence
    assert trtre_backend_mock._context.execute_async_v3.call_count == 2  # Setup + capture
    mock_graph.capture_begin.assert_called_once()
    mock_graph.capture_end.assert_called_once()
    mock_graph.replay.assert_called_once()

    # Check that _infer_cuda_graph pointer changed from _build_cuda_graph to _execute_cuda_graph
    assert trtre_backend_mock._infer_cuda_graph == trtre_backend_mock._execute_cuda_graph


@requires_cuda
def test_execute_with_cuda_graphs_subsequent_run(trtre_backend_mock):
    """Test CUDA graph execution on subsequent runs (launch phase)."""
    # Set up existing CUDA graph
    trtre_backend_mock._cuda_graph = Mock()

    # Execute
    trtre_backend_mock._execute_cuda_graph()

    # Verify launch
    trtre_backend_mock._cuda_graph.replay.assert_called_once()


@requires_cuda
def test_execute_with_cuda_graphs_capture_failure(trtre_backend_mock, mocker):
    """Test CUDA graph execution handles capture failure."""

    mock_graph = Mock()
    mocker.patch("torch.cuda.CUDAGraph", return_value=mock_graph)

    # Mock successful setup but failed capture
    def mock_execute(*_args, **_kwargs):
        if trtre_backend_mock._context.execute_async_v3.call_count == 1:
            return True  # Setup succeeds
        else:
            return False  # Capture fails

    trtre_backend_mock._context.execute_async_v3.side_effect = mock_execute

    # Should raise RuntimeError
    with pytest.raises(RuntimeError, match="TensorRT execution failed during CUDA graph capture"):
        trtre_backend_mock._build_cuda_graph()


@requires_cuda
def test_describe_with_cuda_graphs():
    """Test backend description includes CUDA graphs setting."""
    config = TensorRTBackendConfig(use_cuda_graphs=True)
    backend = TensorRTBackend(config=config)

    description = backend.describe()
    assert "use_cuda_graphs=True" in description or "use_cuda_graphs: True" in description
